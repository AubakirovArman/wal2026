"""WAL-TAT: transactional, causally ranked ternary conversion."""

from .campaign import (
    accepted_weight_counts,
    atomic_write_json,
    coverage_proportional_nll_gate,
    validate_checkpoint_deletion_target,
    worst_ratio,
)
from .controller import (
    AdaptiveTransactionSizer,
    GateDecision,
    RatioGate,
    TransactionController,
    TransactionSizeDecision,
)
from .compensation import (
    linked_down_group_mask,
    linked_gqa_output_group_mask,
    linked_gqa_query_group_mask,
    linked_output_group_mask,
    structured_channel_candidate_mask,
    structured_down_candidate_mask,
)
from .evaluation import LossMetrics, evaluate_causal_lm
from .moments import CausalMomentCollector
from .quantization import (
    hard_codes_scales,
    hestia_quantize,
    initial_group_scales,
    q2_g128_physical_bpw,
    transaction_schedule,
)
from .proxy import ProxyTernaryLinear, ProxyTernaryMatrix, soft_ternary_proxy
from .orchestration import (
    active_campaign_workers,
    common_campaign_frontier,
    load_campaign_state,
    sha256_file,
    synchronize_campaign_frontiers,
)
from .scoring import (
    activation_fisher_group_damage,
    diagonal_ternary_search,
    group_weight_mse,
    select_group_mask,
    sensitivity_decile_mask,
)
from .transaction import (
    AtomicTernaryTransaction,
    TransactionalTernaryLinear,
    TransactionalTernaryMatrix,
)
from .transforms import (
    FixedTernaryLinear,
    TransformedProxyTernaryLinear,
    blockwise_randomized_hadamard,
    inverse_blockwise_randomized_hadamard,
    normalized_hadamard,
    rademacher_signs,
)
from .uncertainty import paired_block_bootstrap_nll
from .wal import HashChainWAL, WALIntegrityError, WALRecord

__all__ = [
    "FixedTernaryLinear",
    "TransformedProxyTernaryLinear",
    "blockwise_randomized_hadamard",
    "inverse_blockwise_randomized_hadamard",
    "normalized_hadamard",
    "rademacher_signs",
    "CausalMomentCollector",
    "AdaptiveTransactionSizer",
    "AtomicTernaryTransaction",
    "GateDecision",
    "HashChainWAL",
    "LossMetrics",
    "RatioGate",
    "TransactionController",
    "TransactionSizeDecision",
    "TransactionalTernaryLinear",
    "TransactionalTernaryMatrix",
    "WALIntegrityError",
    "WALRecord",
    "activation_fisher_group_damage",
    "accepted_weight_counts",
    "atomic_write_json",
    "coverage_proportional_nll_gate",
    "diagonal_ternary_search",
    "evaluate_causal_lm",
    "group_weight_mse",
    "hard_codes_scales",
    "hestia_quantize",
    "initial_group_scales",
    "linked_down_group_mask",
    "linked_gqa_output_group_mask",
    "linked_gqa_query_group_mask",
    "linked_output_group_mask",
    "q2_g128_physical_bpw",
    "paired_block_bootstrap_nll",
    "ProxyTernaryLinear",
    "ProxyTernaryMatrix",
    "soft_ternary_proxy",
    "active_campaign_workers",
    "common_campaign_frontier",
    "load_campaign_state",
    "sha256_file",
    "synchronize_campaign_frontiers",
    "select_group_mask",
    "sensitivity_decile_mask",
    "structured_channel_candidate_mask",
    "structured_down_candidate_mask",
    "transaction_schedule",
    "validate_checkpoint_deletion_target",
    "worst_ratio",
]
