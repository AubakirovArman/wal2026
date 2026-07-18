"""Matched up-only vs atomic linked up/down WAL-TAT experiment on Qwen3-1.7B."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wal_tat import (
    AtomicTernaryTransaction,
    CausalMomentCollector,
    HashChainWAL,
    TransactionalTernaryLinear,
    TransactionalTernaryMatrix,
    activation_fisher_group_damage,
    linked_down_group_mask,
    linked_output_group_mask,
    structured_channel_candidate_mask,
    transaction_schedule,
)


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
UP_NAME = "model.layers.27.mlp.up_proj"
DOWN_NAME = "model.layers.27.mlp.down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=WORKSPACE / "wal2/checkpoints/wal-tat-compose-down-full-up-frontier.pt",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-suite-v1-l128-wc4-cc4-g8.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--candidate-name", default=UP_NAME)
    parser.add_argument("--tag", default="compensation_window_v1")
    parser.add_argument("--fraction", type=float, default=0.015625)
    parser.add_argument("--channel-blocks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--kd-weight", type=float, default=0.1)
    parser.add_argument("--kd-topk", type=int, default=64)
    parser.add_argument("--kd-stride", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--hidden-kd-weight", type=float, default=0.0)
    parser.add_argument(
        "--hidden-module",
        default="model.layers.27.mlp",
        help="module whose output is matched when hidden KD is enabled",
    )
    parser.add_argument(
        "--teacher-source",
        choices=("frontier", "bf16"),
        default="frontier",
        help="distill from the source quantized frontier or the original BF16 model",
    )
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--arms", default="candidate_only,linked_down")
    parser.add_argument(
        "--scale-only-compensation",
        action="store_true",
        help="train only scales, not master weights, in reopened compensation matrices",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    data = value.detach().to("cpu", dtype=torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def default_model_path() -> Path:
    snapshots = sorted(
        (WORKSPACE / ".hf_cache/models--Qwen--Qwen3-1.7B/snapshots").iterdir()
    )
    if len(snapshots) != 1:
        raise RuntimeError(f"expected one local Qwen3-1.7B snapshot, found {snapshots}")
    return snapshots[0].resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


@torch.no_grad()
def install_checkpoint(model, payload: Mapping, device: str):
    if payload.get("format") not in {"wal-tat-multi-g128-v1", "wal-tat-multi-g128-v2"}:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    matrices = {}
    for name, entry in payload["matrices"].items():
        target = model.get_submodule(name)
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{name} is not a linear layer")
        matrix = TransactionalTernaryMatrix(
            target.weight, group_size=int(entry["group_size"])
        ).to(device)
        matrix.master_weight.copy_(entry["fp_master_bf16"].to(device).float())
        matrix.group_scale.copy_(entry["scales_fp16"].to(device).float())
        matrix.committed_mask.copy_(entry["committed_mask"].to(device))
        codes = entry["ternary_codes_int8"].to(device, dtype=torch.int8)
        if matrix.padding:
            codes = F.pad(codes, (0, matrix.padding))
        matrix.committed_codes.copy_(codes.view_as(matrix.committed_codes))
        wrapper = TransactionalTernaryLinear(matrix, target.bias).to(device)
        set_submodule(model, name, wrapper)
        matrices[name] = matrix
    parameters = dict(model.named_parameters())
    for name, value in payload.get("norm_extras", {}).items():
        parameter = parameters.get(name)
        if parameter is not None and tuple(parameter.shape) == tuple(value.shape):
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    return matrices


@torch.no_grad()
def ensure_candidate_matrix(model, matrices, name: str, device: str, group_size: int = 128):
    if name in matrices:
        return matrices[name]
    target = model.get_submodule(name)
    if not isinstance(target, nn.Linear):
        raise TypeError(f"{name} is not a linear layer")
    matrix = TransactionalTernaryMatrix(target.weight, group_size=group_size).to(device)
    set_submodule(model, name, TransactionalTernaryLinear(matrix, target.bias).to(device))
    matrices[name] = matrix
    return matrix


@torch.no_grad()
def evaluate_chunks(model, chunks: Sequence[torch.Tensor], device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for chunk in chunks:
        batch = chunk.unsqueeze(0).to(device)
        logits = model(input_ids=batch[:, :-1], use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += batch.shape[1] - 1
    nll = total_loss / max(total_tokens, 1)
    return {"nll": nll, "ppl": math.exp(min(nll, 50.0)), "tokens": total_tokens}


def evaluate_domains(model, gates, device: str):
    return {name: evaluate_chunks(model, chunks, device) for name, chunks in gates.items()}


def ratios(metrics, baseline):
    return {
        domain: metrics[domain]["nll"] / baseline[domain]["nll"]
        for domain in baseline
    }


def teacher_bucket(logits: torch.Tensor, args: argparse.Namespace):
    scaled = logits[:, :: args.kd_stride].float() / args.kd_temperature
    values, indices = scaled.topk(min(args.kd_topk, scaled.shape[-1]), dim=-1)
    logz = torch.logsumexp(scaled, dim=-1, keepdim=True)
    top_logp = values - logz
    other = (1.0 - top_logp.exp().sum(-1)).clamp_min(1e-7)
    return {
        "indices": indices.squeeze(0).cpu(),
        "top_logp": top_logp.squeeze(0).half().cpu(),
        "other": other.squeeze(0).half().cpu(),
    }


def tensor_output(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("hidden KD module must return a tensor or tensor-first tuple")


@torch.no_grad()
def cache_teacher(model, calibration, args):
    result = []
    hidden = {}
    handle = None
    if args.hidden_kd_weight > 0:
        module = model.get_submodule(args.hidden_module)
        handle = module.register_forward_hook(
            lambda _module, _inputs, output: hidden.__setitem__("value", tensor_output(output))
        )
    try:
        for chunk in calibration:
            hidden.clear()
            batch = chunk.unsqueeze(0).to(args.device)
            bucket = teacher_bucket(
                model(input_ids=batch[:, :-1], use_cache=False).logits, args
            )
            if args.hidden_kd_weight > 0:
                bucket["hidden"] = hidden["value"].squeeze(0).half().cpu()
            result.append(bucket)
    finally:
        if handle is not None:
            handle.remove()
    return result


def bucket_kl(student_logits: torch.Tensor, cache, args):
    scaled = student_logits[:, :: args.kd_stride].float() / args.kd_temperature
    indices = cache["indices"].to(scaled.device).unsqueeze(0)
    teacher_top_logp = cache["top_logp"].to(scaled.device).float().unsqueeze(0)
    teacher_other = cache["other"].to(scaled.device).float().unsqueeze(0).clamp_min(1e-7)
    student_logz = torch.logsumexp(scaled, dim=-1, keepdim=True)
    student_top_logp = scaled.gather(-1, indices) - student_logz
    teacher_top_p = teacher_top_logp.exp()
    top_term = (teacher_top_p * (teacher_top_logp - student_top_logp)).sum(-1)
    student_other = (1.0 - student_top_logp.exp().sum(-1)).clamp_min(1e-7)
    other_term = teacher_other * (teacher_other.log() - student_other.log())
    return ((top_term + other_term) * args.kd_temperature**2).mean()


def adaptive_candidate_scores(model, matrix, candidate_name, calibration, args):
    eligible = ~matrix.committed_mask
    matrix.begin(eligible, transaction_id="measure-up-frontier")
    matrix.set_candidate_state(0.0, 0.3508855606815209)
    matrix.master_weight.requires_grad_(True)
    target = model.get_submodule(candidate_name)
    with CausalMomentCollector(target) as collector:
        for index, chunk in enumerate(calibration):
            model.zero_grad(set_to_none=True)
            batch = chunk.unsqueeze(0).to(args.device)
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1)
            )
            loss.backward()
            log(f"moment {index + 1}/{len(calibration)} loss={loss.item():.4f}")
        input_moment, output_gradient_moment = collector.moments()
    model.zero_grad(set_to_none=True)
    matrix.rollback()
    matrix.master_weight.requires_grad_(False)
    scores = activation_fisher_group_damage(
        matrix.master_weight,
        input_moment,
        output_gradient_moment,
        group_size=matrix.group_size,
        scales=matrix.group_scale,
        relative=True,
    )
    scores[matrix.committed_mask] = float("inf")
    return scores


def compensation_parameters(model):
    values = list(model.model.norm.parameters())
    values += list(model.model.layers[27].post_attention_layernorm.parameters())
    unique, seen = [], set()
    for parameter in values:
        if id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


@torch.no_grad()
def scale_change_stats(matrix, before: torch.Tensor, mask: torch.Tensor):
    delta = matrix.group_scale.detach() - before
    relative = delta.abs() / before.abs().clamp_min(1e-8)
    selected = relative[mask]
    absolute = delta.abs()[mask]
    return {
        "changed_groups": int((delta[mask] != 0).sum().item()),
        "groups": int(mask.sum().item()),
        "absolute_mean": float(absolute.mean().item()),
        "relative_mean": float(selected.mean().item()),
        "relative_p95": float(torch.quantile(selected, 0.95).item()),
        "relative_max": float(selected.max().item()),
    }


def train_atomic(model, atomic, matrices, calibration, teacher, extras, args, wal):
    parameters = []
    for name, matrix in matrices.items():
        if name == "candidate" or not args.scale_only_compensation:
            parameters.append(matrix.master_weight)
        parameters.append(matrix.group_scale)
    parameters.extend(extras)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    history = []
    student_hidden = {}
    hidden_handle = None
    if args.hidden_kd_weight > 0:
        module = model.get_submodule(args.hidden_module)
        hidden_handle = module.register_forward_hook(
            lambda _module, _inputs, output: student_hidden.__setitem__(
                "value", tensor_output(output)
            )
        )
    model.train()
    try:
        for step in range(1, args.steps + 1):
            pressure, temperature = transaction_schedule(step, args.steps)
            atomic.set_candidate_state(pressure, temperature)
            item = (step - 1) % len(calibration)
            batch = calibration[item].unsqueeze(0).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            student_hidden.clear()
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1)
            )
            kd = bucket_kl(logits, teacher[item], args)
            hidden_kd = torch.zeros((), device=logits.device)
            if args.hidden_kd_weight > 0:
                student = student_hidden["value"].float()
                target = teacher[item]["hidden"].to(student.device).float().unsqueeze(0)
                hidden_kd = F.mse_loss(student, target) / target.square().mean().clamp_min(1e-8)
            loss = ce + args.kd_weight * kd + args.hidden_kd_weight * hidden_kd
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step == 1 or step == args.steps or step % max(1, args.steps // 4) == 0:
                entry = {
                    "step": step,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "kd": float(kd.item()),
                    "hidden_kd": float(hidden_kd.item()),
                    "pressure": pressure,
                    "temperature": temperature,
                    "gradient_norm": float(gradient_norm),
                    "code_churn": atomic.current_code_churn(),
                }
                history.append(entry)
                wal.append("progress", atomic.transaction_id, entry)
                log(
                    f"step={step}/{args.steps} loss={loss.item():.4f} ce={ce.item():.4f} "
                    f"kd={kd.item():.4f} hidden={hidden_kd.item():.4f} "
                    f"pressure={pressure:.3f}"
                )
    finally:
        if hidden_handle is not None:
            hidden_handle.remove()
    atomic.set_candidate_state(1.0, 0.0)
    model.eval()
    return history


@torch.no_grad()
def checkpoint_payload(model, matrices, parent_hash, metrics, metric_ratios, metadata):
    entries = {}
    for name, matrix in matrices.items():
        codes, scales = matrix.codes_scales()
        entries[name] = {
            "shape": tuple(matrix.master_weight.shape),
            "group_size": matrix.group_size,
            "committed_mask": matrix.committed_mask.cpu(),
            "ternary_codes_int8": codes.cpu(),
            "scales_fp16": scales.half().cpu(),
            "fp_master_bf16": matrix.master_weight.detach().bfloat16().cpu(),
        }
    return {
        "format": "wal-tat-multi-g128-v2",
        "parent_sha256": parent_hash,
        "matrices": entries,
        "norm_extras": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if "norm" in name and ".matrix." not in name
        },
        "metrics": metrics,
        "gate_ratios": metric_ratios,
        "metadata": metadata,
    }


def run_arm(
    arm,
    args,
    model_path,
    source_payload,
    source_hash,
    calibration,
    gates,
    baseline,
    candidate_mask_cpu,
    down_mask_cpu,
    up_mask_cpu,
    selected_blocks,
    candidate_name,
    wal,
    checkpoint_path,
):
    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    teacher = (
        cache_teacher(model, calibration, args)
        if args.teacher_source == "bf16"
        else None
    )
    matrices = install_checkpoint(model, source_payload, args.device)
    candidate_matrix = ensure_candidate_matrix(
        model, matrices, candidate_name, args.device
    )
    down_matrix = matrices[DOWN_NAME]
    starting = evaluate_domains(model, gates, args.device)
    if teacher is None:
        teacher = cache_teacher(model, calibration, args)
    extras = compensation_parameters(model)
    extra_snapshots = [parameter.detach().clone() for parameter in extras]
    candidate_mask = candidate_mask_cpu.to(args.device)
    down_mask = down_mask_cpu.to(args.device)
    up_mask = up_mask_cpu.to(args.device)
    active = {"candidate": candidate_matrix}
    masks = {"candidate": candidate_mask}
    reopen = set()
    if arm == "linked_down":
        active["down"] = down_matrix
        masks["down"] = down_mask
        reopen.add("down")
    elif arm == "linked_mlp":
        if candidate_name == UP_NAME:
            raise ValueError("linked_mlp requires a gate projection candidate")
        active["up"] = matrices[UP_NAME]
        masks["up"] = up_mask
        reopen.add("up")
        active["down"] = down_matrix
        masks["down"] = down_mask
        reopen.add("down")
    scale_before = {
        name: matrix.group_scale.detach().clone() for name, matrix in active.items()
    }
    atomic = AtomicTernaryTransaction(active)
    transaction_id = f"{args.tag}-{arm}"
    atomic.begin(masks, reopen=reopen, transaction_id=transaction_id)
    wal.append(
        "begin",
        transaction_id,
        {
            "arm": arm,
            "candidate_name": candidate_name,
            "candidate_groups": int(candidate_mask.sum().item()),
            "down_reopened_groups": (
                int(down_mask.sum().item())
                if arm in {"linked_down", "linked_mlp"}
                else 0
            ),
            "up_reopened_groups": int(up_mask.sum().item()) if arm == "linked_mlp" else 0,
            "scale_only_compensation": args.scale_only_compensation,
            "teacher_source": args.teacher_source,
            "hidden_kd_weight": args.hidden_kd_weight,
            "hidden_module": args.hidden_module,
            "selected_channel_blocks": list(selected_blocks),
            "candidate_mask_sha256": tensor_digest(candidate_mask),
            "down_mask_sha256": tensor_digest(down_mask) if arm == "linked_down" else None,
            "steps": args.steps,
            "lr": args.lr,
        },
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = train_atomic(
        model, atomic, active, calibration, teacher, extras, args, wal
    )
    attempted = evaluate_domains(model, gates, args.device)
    attempted_ratios = ratios(attempted, baseline)
    scale_changes = {
        name: scale_change_stats(matrix, scale_before[name], masks[name])
        for name, matrix in active.items()
    }
    passed = all(value <= args.gate_ratio for value in attempted_ratios.values())
    intent = "commit_intent" if passed else "rollback_intent"
    wal.append(
        intent,
        transaction_id,
        {"arm": arm, "metrics": attempted, "ratios": attempted_ratios},
    )
    churn = atomic.current_code_churn()
    if passed:
        atomic_result = atomic.commit()
        terminal = "commit"
    else:
        atomic_result = atomic.rollback()
        with torch.no_grad():
            for parameter, snapshot in zip(extras, extra_snapshots):
                parameter.copy_(snapshot)
        terminal = "rollback"
    for parameter in extras:
        parameter.requires_grad_(False)
    deployed = evaluate_domains(model, gates, args.device)
    deployed_ratios = ratios(deployed, baseline)
    elapsed = time.time() - started
    peak = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else None
    )
    wal.append(
        terminal,
        transaction_id,
        {
            "arm": arm,
            "ratios": deployed_ratios,
            "code_churn": churn,
            "matrix_result": atomic_result,
        },
    )
    result = {
        "arm": arm,
        "passed": passed,
        "starting": starting,
        "attempted": attempted,
        "attempted_ratios": attempted_ratios,
        "deployed": deployed,
        "deployed_ratios": deployed_ratios,
        "code_churn": churn,
        "scale_changes": scale_changes,
        "history": history,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak,
        "coverage": {
            name: float(matrix.committed_mask.float().mean().item())
            for name, matrix in matrices.items()
        },
        "projected_bpw": {
            name: matrix.projected_mixed_bpw() for name, matrix in matrices.items()
        },
    }
    if passed:
        payload = checkpoint_payload(
            model,
            matrices,
            source_hash,
            deployed,
            deployed_ratios,
            {
                "experiment": args.tag,
                "arm": arm,
                "candidate_name": candidate_name,
                "candidate_new_groups": int(candidate_mask.sum().item()),
                "down_reopened_groups": (
                    int(down_mask.sum().item())
                    if arm in {"linked_down", "linked_mlp"}
                    else 0
                ),
                "up_reopened_groups": int(up_mask.sum().item()) if arm == "linked_mlp" else 0,
                "scale_only_compensation": args.scale_only_compensation,
                "teacher_source": args.teacher_source,
                "hidden_kd_weight": args.hidden_kd_weight,
                "hidden_module": args.hidden_module,
                "selected_channel_blocks": list(selected_blocks),
                "steps": args.steps,
                "lr": args.lr,
            },
        )
        torch.save(payload, checkpoint_path)
        result["checkpoint"] = str(checkpoint_path)
    del teacher, model, matrices, candidate_matrix, down_matrix
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def fresh_verify(checkpoint, model_path, suite, args):
    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, suite["gates"], args.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    matrices = install_checkpoint(model, payload, args.device)
    metrics = evaluate_domains(model, suite["gates"], args.device)
    metric_ratios = ratios(metrics, baseline)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "baseline": baseline,
        "metrics": metrics,
        "ratios": metric_ratios,
        "passed": all(value <= args.gate_ratio for value in metric_ratios.values()),
        "coverage": {
            name: float(matrix.committed_mask.float().mean().item())
            for name, matrix in matrices.items()
        },
        "codes": {
            name: sorted(
                matrix.committed_codes[matrix.committed_mask].unique().cpu().tolist()
            )
            for name, matrix in matrices.items()
        },
    }
    del model, matrices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_report(path: Path, result: Mapping) -> None:
    lines = [
        "# WAL-TAT atomic compensation window",
        "",
        f"Candidate: `{result['candidate_name']}`; new groups: "
        f"{result['selection']['new_candidate_groups']}; linked down groups: "
        f"{result['selection']['linked_down_groups']}.",
        "",
        "| Arm | Gate | Wiki ratio | Code ratio | Candidate coverage | Down churn |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, value in result["arms"].items():
        lines.append(
            f"| {arm} | {'commit' if value['passed'] else 'rollback'} | "
            f"{value['attempted_ratios']['wiki']:.6f} | "
            f"{value['attempted_ratios']['code']:.6f} | "
            f"{value['coverage'][result['candidate_name']]:.2%} | "
            f"{value['code_churn'].get('down', 0.0):.6f} |"
        )
    if result.get("fresh_verify"):
        verify = result["fresh_verify"]
        lines += [
            "",
            f"Fresh verify: {'pass' if verify['passed'] else 'fail'}; Wiki/code ratios "
            f"{verify['ratios']['wiki']:.6f}/{verify['ratios']['code']:.6f}.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if any(value not in {"candidate_only", "up_only", "linked_down", "linked_mlp"} for value in arms):
        raise ValueError(f"unsupported arms: {arms}")
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(args.source_checkpoint)
    suite_hash = sha256_file(args.suite)
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration, gates = suite["calibration"], suite["gates"]
    seed_everything(args.seed)
    log(f"device={args.device} model={model_path.name} preparing matched selection")

    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    matrices = install_checkpoint(model, source_payload, args.device)
    candidate_matrix = ensure_candidate_matrix(
        model, matrices, args.candidate_name, args.device
    )
    starting = evaluate_domains(model, gates, args.device)
    starting_ratios = ratios(starting, baseline)
    scores = adaptive_candidate_scores(
        model, candidate_matrix, args.candidate_name, calibration, args
    )
    count = max(1, round(scores.numel() * args.fraction))
    candidate_mask, selected_blocks = structured_channel_candidate_mask(
        scores,
        ~candidate_matrix.committed_mask,
        count=count,
        channel_block_size=candidate_matrix.group_size,
        block_count=args.channel_blocks,
    )
    down_mask = linked_down_group_mask(matrices[DOWN_NAME].committed_mask, selected_blocks)
    up_mask = linked_output_group_mask(
        matrices[UP_NAME].committed_mask,
        selected_blocks,
        channel_block_size=candidate_matrix.group_size,
    )
    selection = {
        "new_candidate_groups": int(candidate_mask.sum().item()),
        "new_candidate_fraction": float(candidate_mask.float().mean().item()),
        "linked_down_groups": int(down_mask.sum().item()),
        "linked_down_fraction": float(down_mask.float().mean().item()),
        "linked_up_groups": int(up_mask.sum().item()),
        "linked_up_fraction": float(up_mask.float().mean().item()),
        "selected_channel_blocks": list(selected_blocks),
        "selected_score_mean": float(scores[candidate_mask].mean().item()),
        "selected_score_p95": float(torch.quantile(scores[candidate_mask], 0.95).item()),
        "candidate_mask_sha256": tensor_digest(candidate_mask),
        "down_mask_sha256": tensor_digest(down_mask),
    }
    log(
        f"baseline ratios frontier={starting_ratios}; candidate={args.candidate_name} "
        f"selected={selection['new_candidate_groups']} "
        f"linked_down={selection['linked_down_groups']} blocks={selected_blocks}"
    )
    candidate_mask_cpu, down_mask_cpu, up_mask_cpu = (
        candidate_mask.cpu(),
        down_mask.cpu(),
        up_mask.cpu(),
    )
    del scores, model, matrices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_path = PROJECT / f"results/{args.tag}.json"
    report_path = PROJECT / f"results/{args.tag}.md"
    wal_path = PROJECT / f"results/{args.tag}.wal.jsonl"
    if wal_path.exists():
        wal_path.unlink()
    wal = HashChainWAL(wal_path)
    arm_results = {}
    candidate_paths = {}
    for arm in arms:
        log(f"starting matched arm={arm}")
        candidate_path = (
            WORKSPACE / f"wal2/checkpoints/wal-tat-{args.tag}-{arm}.pt"
        )
        arm_results[arm] = run_arm(
            arm,
            args,
            model_path,
            source_payload,
            source_hash,
            calibration,
            gates,
            baseline,
            candidate_mask_cpu,
            down_mask_cpu,
            up_mask_cpu,
            selected_blocks,
            args.candidate_name,
            wal,
            candidate_path,
        )
        if arm_results[arm]["passed"]:
            candidate_paths[arm] = candidate_path
        log(f"arm={arm} pass={arm_results[arm]['passed']} ratios={arm_results[arm]['attempted_ratios']}")

    chosen_arm = next(
        (
            name
            for name in ("linked_mlp", "linked_down", "candidate_only", "up_only")
            if name in candidate_paths
        ),
        None,
    )
    fresh = fresh_verify(candidate_paths[chosen_arm], model_path, suite, args) if chosen_arm else None
    result = {
        "schema": "wal-tat-compensation-window-v1",
        "tag": args.tag,
        "model": str(model_path),
        "source_checkpoint": str(args.source_checkpoint),
        "source_sha256": source_hash,
        "suite": str(args.suite),
        "suite_sha256": suite_hash,
        "seed": args.seed,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "steps": args.steps,
        "lr": args.lr,
        "gate_ratio": args.gate_ratio,
        "teacher_source": args.teacher_source,
        "kd_weight": args.kd_weight,
        "hidden_kd_weight": args.hidden_kd_weight,
        "hidden_module": args.hidden_module,
        "candidate_name": args.candidate_name,
        "baseline": baseline,
        "starting": starting,
        "starting_ratios": starting_ratios,
        "selection": selection,
        "arms": arm_results,
        "chosen_arm": chosen_arm,
        "fresh_verify": fresh,
        "wal": str(wal_path),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, result)
    log(f"wrote {result_path} and {report_path}; chosen={chosen_arm}")


if __name__ == "__main__":
    main()
