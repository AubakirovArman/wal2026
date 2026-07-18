"""WAL-TAT: transactional, causally ranked ternary conversion."""

from .controller import GateDecision, RatioGate, TransactionController
from .evaluation import LossMetrics, evaluate_causal_lm
from .moments import CausalMomentCollector
from .quantization import (
    hard_codes_scales,
    hestia_quantize,
    initial_group_scales,
    q2_g128_physical_bpw,
    transaction_schedule,
)
from .scoring import (
    activation_fisher_group_damage,
    diagonal_ternary_search,
    group_weight_mse,
    select_group_mask,
)
from .transaction import TransactionalTernaryLinear, TransactionalTernaryMatrix
from .wal import HashChainWAL, WALIntegrityError, WALRecord

__all__ = [
    "CausalMomentCollector",
    "GateDecision",
    "HashChainWAL",
    "LossMetrics",
    "RatioGate",
    "TransactionController",
    "TransactionalTernaryLinear",
    "TransactionalTernaryMatrix",
    "WALIntegrityError",
    "WALRecord",
    "activation_fisher_group_damage",
    "diagonal_ternary_search",
    "evaluate_causal_lm",
    "group_weight_mse",
    "hard_codes_scales",
    "hestia_quantize",
    "initial_group_scales",
    "q2_g128_physical_bpw",
    "select_group_mask",
    "transaction_schedule",
]
