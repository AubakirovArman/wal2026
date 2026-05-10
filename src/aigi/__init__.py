"""AIGI pre-alpha SDK.

AIGI is a verified memory-accumulation layer built on top of WAL concepts.
This package intentionally exposes a conservative MVP: verified memory
candidates, tier selection, commit/rollback semantics, and audit logs.
"""

from aigi.core.state import AIGIResponse, CompileReport, MemoryCandidate, MemoryPolicy
from aigi.core.system import AIGISystem
from aigi.governance.budget import BudgetDecision, MemoryBudgetEvaluator, MemoryChangeBudget
from aigi.governance.report import CommitDecisionReport, CommitDecisionReporter
from aigi.governance.risk import RiskLedger
from aigi.learn.experience import Experience, Lesson, LessonExtractor
from aigi.learn.loop import LearningLoopResult, VerifiedLearningLoop
from aigi.model import (
    HuggingFaceTextBackend,
    LogitLoRAAdapterTrainer,
    LogitLoRATrainingReport,
    ModuleLoRAAdapterTrainer,
    ModuleLoRATrainingReport,
    SoftPromptAdapterTrainer,
    SoftPromptTrainingReport,
    StaticTextModelBackend,
    TextModelBackend,
)
from aigi.verify.contracts import BehavioralContract, BehavioralContractVerifier, ContractExpectation
from aigi.verify.regression import ContractRegressionSuite, RegressionSuiteResult

__all__ = [
    "AIGISystem",
    "AIGIResponse",
    "BehavioralContract",
    "BehavioralContractVerifier",
    "BudgetDecision",
    "CompileReport",
    "CommitDecisionReport",
    "CommitDecisionReporter",
    "ContractRegressionSuite",
    "ContractExpectation",
    "Experience",
    "HuggingFaceTextBackend",
    "LogitLoRAAdapterTrainer",
    "LogitLoRATrainingReport",
    "ModuleLoRAAdapterTrainer",
    "ModuleLoRATrainingReport",
    "Lesson",
    "LessonExtractor",
    "LearningLoopResult",
    "MemoryBudgetEvaluator",
    "MemoryCandidate",
    "MemoryChangeBudget",
    "MemoryPolicy",
    "RegressionSuiteResult",
    "RiskLedger",
    "SoftPromptAdapterTrainer",
    "SoftPromptTrainingReport",
    "StaticTextModelBackend",
    "TextModelBackend",
    "VerifiedLearningLoop",
]
