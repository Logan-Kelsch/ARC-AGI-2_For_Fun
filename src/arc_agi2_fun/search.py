from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .data import ArcTask
from .programs import GridProgram, PROGRAM_LIBRARY
from .scoring import cell_accuracy, grid_exact


@dataclass(frozen=True)
class CandidateScore:
    program: GridProgram
    exact_pairs: int
    total_pairs: int
    mean_cell_accuracy: float

    @property
    def exact_fraction(self) -> float:
        return self.exact_pairs / self.total_pairs if self.total_pairs else 0.0

    @property
    def rank_key(self) -> tuple[float, float, int]:
        return (
            self.exact_fraction,
            self.mean_cell_accuracy,
            -self.program.complexity,
        )


def score_program(task: ArcTask, program: GridProgram) -> CandidateScore:
    exact = 0
    diagnostic: list[float] = []
    for pair in task.train:
        prediction = program.apply(pair.input)
        exact += int(grid_exact(prediction, pair.output))
        diagnostic.append(cell_accuracy(prediction, pair.output))
    return CandidateScore(
        program=program,
        exact_pairs=exact,
        total_pairs=len(task.train),
        mean_cell_accuracy=mean(diagnostic) if diagnostic else 0.0,
    )


def rank_programs(
    task: ArcTask,
    programs: tuple[GridProgram, ...] = PROGRAM_LIBRARY,
) -> list[CandidateScore]:
    scored = [score_program(task, program) for program in programs]
    return sorted(scored, key=lambda item: item.rank_key, reverse=True)
