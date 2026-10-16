"""Expose plans and instruction emitters that are independent of model family.

A plan fixes memory locations and layouts; a lowerer emits commands for that
plan. Some mathematical operations remain partial, as documented by their
individual lowerers.
"""

from .bindings import (
    CentDramRowRange,
    CentDramVector,
    CentSharedBufferSpan,
    CentSharedBufferVector,
)
from .data_movement import (
    CentBankGroupVectorTransferPlan,
    lower_load_bank_group_vector,
    lower_store_bank_group_vector,
)
from .elementwise import CentAccumulatePlan, lower_accumulate
from .linear import CentWeightGemvPlan, lower_weight_gemv, plan_weight_gemv
from .normalization import (
    CentL2NormPlan,
    CentRmsNormPlan,
    CentSumOfSquaresPlan,
    lower_l2_norm,
    lower_rms_norm,
    lower_sum_of_squares,
)
from .planning import (
    CentPartitionedVectorLayout,
    pack_zero_padded_vector,
    plan_partitioned_vector,
)

__all__ = [
    "CentAccumulatePlan",
    "CentBankGroupVectorTransferPlan",
    "CentDramRowRange",
    "CentDramVector",
    "CentL2NormPlan",
    "CentPartitionedVectorLayout",
    "CentRmsNormPlan",
    "CentSharedBufferSpan",
    "CentSharedBufferVector",
    "CentSumOfSquaresPlan",
    "CentWeightGemvPlan",
    "lower_accumulate",
    "lower_l2_norm",
    "lower_load_bank_group_vector",
    "lower_rms_norm",
    "lower_store_bank_group_vector",
    "lower_sum_of_squares",
    "lower_weight_gemv",
    "pack_zero_padded_vector",
    "plan_partitioned_vector",
    "plan_weight_gemv",
]
