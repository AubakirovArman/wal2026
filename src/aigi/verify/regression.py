from __future__ import annotations

from dataclasses import dataclass

from aigi.core.state import GateResult
from aigi.verify.contracts import BehavioralContract, BehavioralContractVerifier


@dataclass(frozen=True)
class RegressionSuiteResult:
    status: str
    gates: tuple[GateResult, ...]

    @property
    def pass_(self) -> bool:
        return self.status == "PASS"

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


class ContractRegressionSuite:
    def __init__(self, contracts: tuple[BehavioralContract, ...]) -> None:
        self.contracts = contracts
        self.verifier = BehavioralContractVerifier()

    @classmethod
    def from_contract(cls, contract: BehavioralContract) -> "ContractRegressionSuite":
        return cls((contract,))

    def evaluate_system(self, ask) -> RegressionSuiteResult:
        gates: list[GateResult] = []
        for contract in self.contracts:
            gates.extend(self.verifier.evaluate_system(ask, contract))
        status = "PASS" if all(gate.passed for gate in gates) else "FAIL"
        return RegressionSuiteResult(status=status, gates=tuple(gates))

