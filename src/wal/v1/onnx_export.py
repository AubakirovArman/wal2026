#!/usr/bin/env python3
"""WAL ONNX Export.

Phase 11b: Export WAL-encoded models to ONNX format.

Two export modes:
- 'simple': Pre-decode WAL weights, export as standard ONNX Linear
- 'native': Export WAL decode as ONNX operations (Gather + Mul + Reshape + Add)
"""
import io
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from .isa import ProgramBufferV1, AtomTableV1, CoeffTable
from .decoder import wal_decode_v1
from .nn import WALLinear, WALCachedLinear


def export_wal_simple(
    model: nn.Module,
    dummy_input: torch.Tensor,
    filepath: Optional[str] = None,
    opset_version: int = 17,
    **kwargs
) -> bytes:
    """Export WAL model to ONNX with pre-decoded weights.
    
    This is the simplest export path: WAL layers are decoded to dense
    weights before export, resulting in a standard ONNX model.
    
    Args:
        model: PyTorch model with WAL layers
        dummy_input: Example input for tracing
        filepath: Output file path (if None, returns bytes)
        opset_version: ONNX opset version
        **kwargs: Additional torch.onnx.export kwargs
    
    Returns:
        ONNX model as bytes (if filepath is None)
    """
    # Create a cloned model with decoded weights
    decoded_model = _build_decoded_model(model)
    decoded_model.eval()
    
    buffer = io.BytesIO()
    torch.onnx.export(
        decoded_model,
        dummy_input,
        buffer,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        **kwargs
    )
    
    onnx_bytes = buffer.getvalue()
    
    if filepath:
        with open(filepath, "wb") as f:
            f.write(onnx_bytes)
    
    return onnx_bytes


def _build_decoded_model(model: nn.Module) -> nn.Module:
    """Create a copy of model with WAL layers replaced by standard Linear."""
    import copy
    decoded = copy.deepcopy(model)
    
    for name, module in list(decoded.named_modules()):
        if isinstance(module, (WALLinear, WALCachedLinear)):
            # Decode weight
            weight = module.wal_weight.decode()
            if weight.shape != module.wal_weight.shape:
                weight = weight.reshape(module.wal_weight.shape)
            
            # Create standard Linear
            linear = nn.Linear(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                bias=module.bias is not None,
                dtype=weight.dtype,
                device=weight.device,
            )
            linear.weight.data = weight
            if module.bias is not None:
                linear.bias.data = module.bias.data
            
            # Replace
            parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(decoded.named_modules())[parent_name]
                setattr(parent, child_name, linear)
            else:
                setattr(decoded, child_name, linear)
    
    return decoded


def export_wal_native(
    wal_layer: WALLinear,
    filepath: Optional[str] = None,
    opset_version: int = 17,
) -> bytes:
    """Export a single WAL layer to ONNX with native WAL decode ops.
    
    The exported graph performs:
    1. Gather atoms by atom_ids
    2. Gather coeffs by coeff_ids
    3. Mul: atom_values * coeff_values
    4. Reshape to weight shape
    5. Add residual (if present)
    6. MatMul with input
    
    Args:
        wal_layer: WAL linear layer
        filepath: Output file path
        opset_version: ONNX opset version
    
    Returns:
        ONNX model as bytes
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("onnx is required. Install with: pip install onnx")
    
    from onnx import helper, TensorProto
    import numpy as np
    
    prog = wal_layer.wal_weight.prog
    atom_table = wal_layer.wal_weight.atom_table
    coeffs = wal_layer.wal_weight.coeffs
    
    # Precompute flat atom values
    from .decoder import precompute_flat_atoms
    flat_atoms = precompute_flat_atoms(atom_table).cpu().numpy().astype(np.float32)
    coeff_values = coeffs.values.cpu().numpy().astype(np.float32)
    
    atom_ids = prog.atom_ids.cpu().numpy().astype(np.int64)
    coeff_ids = prog.coeff_ids.cpu().numpy().astype(np.int64)
    shape = list(wal_layer.wal_weight.shape)
    N = prog.N
    
    # ONNX graph construction
    # Inputs: input tensor [batch, in_features]
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, shape[1]])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, shape[0]])
    
    # Constants
    atoms_init = helper.make_tensor("atoms", TensorProto.FLOAT, flat_atoms.shape, flat_atoms.flatten().tolist())
    coeffs_init = helper.make_tensor("coeffs", TensorProto.FLOAT, coeff_values.shape, coeff_values.flatten().tolist())
    atom_ids_init = helper.make_tensor("atom_ids", TensorProto.INT64, atom_ids.shape, atom_ids.flatten().tolist())
    coeff_ids_init = helper.make_tensor("coeff_ids", TensorProto.INT64, coeff_ids.shape, coeff_ids.flatten().tolist())
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [len(shape)], shape)
    
    nodes = []
    
    # 1. Gather atoms: atom_values = atoms[atom_ids]
    nodes.append(helper.make_node("Gather", ["atoms", "atom_ids"], ["atom_values"], axis=0))
    
    # 2. Gather coeffs: coeff_values = coeffs[coeff_ids]
    nodes.append(helper.make_node("Gather", ["coeffs", "coeff_ids"], ["coeff_values"], axis=0))
    
    # 3. Mul: weight_flat = atom_values * coeff_values
    nodes.append(helper.make_node("Mul", ["atom_values", "coeff_values"], ["weight_flat"]))
    
    # 4. Reshape to weight shape
    nodes.append(helper.make_node("Reshape", ["weight_flat", "shape"], ["weight"]))
    
    # 5. MatMul: output = input @ weight.T
    # ONNX MatMul: [batch, in] @ [in, out] = [batch, out]
    # weight is [out, in], so we need to transpose
    nodes.append(helper.make_node("Transpose", ["weight"], ["weight_t"], perm=[1, 0]))
    nodes.append(helper.make_node("MatMul", ["input", "weight_t"], ["matmul_out"]))
    
    # 6. Add bias if present
    if wal_layer.bias is not None:
        bias = wal_layer.bias.data.cpu().numpy().astype(np.float32)
        bias_init = helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.flatten().tolist())
        nodes.append(helper.make_node("Add", ["matmul_out", "bias"], ["output"]))
        initializers = [atoms_init, coeffs_init, atom_ids_init, coeff_ids_init, shape_init, bias_init]
    else:
        nodes.append(helper.make_node("Identity", ["matmul_out"], ["output"]))
        initializers = [atoms_init, coeffs_init, atom_ids_init, coeff_ids_init, shape_init]
    
    # Residual support
    if prog.residuals.numel() > 0:
        residuals = prog.residuals.cpu().numpy().astype(np.float32)
        resid_init = helper.make_tensor("residuals", TensorProto.FLOAT, residuals.shape, residuals.flatten().tolist())
        # Insert residual add before reshape
        nodes.insert(3, helper.make_node("Add", ["weight_flat", "residuals"], ["weight_flat_resid"]))
        # Update reshape input
        for node in nodes:
            if node.op_type == "Reshape" and node.input[0] == "weight_flat":
                node.input[0] = "weight_flat_resid"
        initializers.append(resid_init)
    
    graph = helper.make_graph(nodes, "wal_native", [input_tensor], [output_tensor], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    model.ir_version = 8
    
    # Validate
    onnx.checker.check_model(model)
    
    onnx_bytes = model.SerializeToString()
    
    if filepath:
        with open(filepath, "wb") as f:
            f.write(onnx_bytes)
    
    return onnx_bytes


def verify_onnx_export(
    wal_layer: WALLinear,
    dummy_input: torch.Tensor,
    onnx_bytes: bytes,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Verify ONNX export matches PyTorch output.
    
    Args:
        wal_layer: WAL layer
        dummy_input: Test input
        onnx_bytes: Exported ONNX model
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for verification")
    
    # PyTorch output
    wal_layer.eval()
    with torch.no_grad():
        pt_out = wal_layer(dummy_input).cpu().numpy()
    
    # ONNX output
    session = ort.InferenceSession(onnx_bytes)
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: dummy_input.cpu().numpy()})[0]
    
    import numpy as np
    close = np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol)
    if not close:
        diff = np.abs(pt_out - onnx_out).max()
        print(f"  Max diff: {diff:.8f}")
    
    return close
