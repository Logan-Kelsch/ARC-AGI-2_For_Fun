from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import combinations, permutations
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .environment import ProgramMeta, ProgramX, SolutionTree, init_env
from .kelschinator import Kelschinator
from .ops import (
    GP_prune,
    OP_REGISTRY,
    SP_generate,
    OperationInfo,
    _eligible_operation_infos,
    _source_tuple,
    _static_source_indices,
    _try_candidate_transactionally,
    gene_atomic_dtypes,
    sample_operation_params,
    valid_generation,
)
from .solve import SolveEvaluationCache
from .wrap import (
    _exact_test_match,
    _load_task_by_id,
    _repo_data_root,
    _solve_frontier,
    _st_progress,
)


@dataclass
class UCTStat:
    """Persistent statistics for one grammar decision node."""

    visits: int = 0
    solve_credit: float = 0.0

    @property
    def exploitation(self) -> float:
        if self.visits <= 0:
            return 0.0
        proportion = min(
            1.0,
            max(0.0, self.solve_credit / float(self.visits)),
        )
        return math.sqrt(proportion)

    def add_visit(self) -> None:
        self.visits += 1

    def add_credit(self, credit: float) -> None:
        if self.visits <= 0:
            return
        self.solve_credit = min(
            float(self.visits),
            self.solve_credit + max(0.0, float(credit)),
        )


@dataclass(frozen=True)
class GrammarSourceKey:
    """Transferable phenotype for an operation/source grammar choice."""

    op_name: str
    source_depth: int
    source_dims: tuple[int, ...]
    source_dtypes: tuple[str, ...]
    source_shapes: tuple[str, ...]


@dataclass
class GrammarUCTPolicy:
    """Cross-task grammar statistics learned by the ARC crawl."""

    gamma: float = 0.85
    exploration_start: float = 8.0
    exploration_end: float = 0.05
    exploration_horizon: int = 1_000_000
    depth_focus: float = 0.08
    total_gene_generations: int = 0
    total_decisions: int = 0
    task_attempts: int = 0
    solved_task_ids: set[str] = field(default_factory=set)
    operation_stats: dict[str, UCTStat] = field(default_factory=dict)
    source_stats: dict[GrammarSourceKey, UCTStat] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if float(self.exploration_start) <= 0.0:
            raise ValueError("exploration_start must be > 0.")
        if float(self.exploration_end) < 0.0:
            raise ValueError("exploration_end must be >= 0.")
        if int(self.exploration_horizon) < 1:
            raise ValueError("exploration_horizon must be >= 1.")
        if float(self.depth_focus) < 0.0:
            raise ValueError("depth_focus must be >= 0.")

        self.gamma = float(self.gamma)
        self.exploration_start = float(self.exploration_start)
        self.exploration_end = float(self.exploration_end)
        self.exploration_horizon = int(self.exploration_horizon)
        self.depth_focus = float(self.depth_focus)
        self.total_gene_generations = int(self.total_gene_generations)
        self.total_decisions = int(self.total_decisions)
        self.task_attempts = int(self.task_attempts)
        self.solved_task_ids = {
            str(task_id)
            for task_id in self.solved_task_ids
        }

    def exploration_coefficient(self) -> float:
        """Exponentially decay start -> end over the configured horizon."""
        if self.total_gene_generations >= self.exploration_horizon:
            return self.exploration_end
        if self.exploration_start == self.exploration_end:
            return self.exploration_end
        if self.exploration_end == 0.0:
            fraction = (
                self.total_gene_generations
                / float(self.exploration_horizon)
            )
            return self.exploration_start * max(0.0, 1.0 - fraction)

        fraction = (
            self.total_gene_generations
            / float(self.exploration_horizon)
        )
        ratio = self.exploration_end / self.exploration_start
        return self.exploration_start * (ratio ** fraction)

    def _uct_score(
        self,
        stat: UCTStat,
        *,
        total_visits: int,
        source_depth: int = 0,
    ) -> tuple[float, float, float]:
        exploitation = stat.exploitation
        coefficient = self.exploration_coefficient()
        exploration = coefficient * math.sqrt(
            math.log(float(total_visits) + 2.0)
            / float(stat.visits + 1)
        )
        depth_bonus = self.depth_focus * (
            float(source_depth) / float(source_depth + 1)
            if source_depth > 0
            else 0.0
        )
        return (
            exploitation + exploration + depth_bonus,
            exploitation,
            exploration,
        )

    def operation_score(self, op_name: str) -> tuple[float, float, float]:
        stat = self.operation_stats.get(op_name, UCTStat())
        return self._uct_score(
            stat,
            total_visits=self.total_decisions,
        )

    def source_score(
        self,
        key: GrammarSourceKey,
    ) -> tuple[float, float, float]:
        stat = self.source_stats.get(key, UCTStat())
        return self._uct_score(
            stat,
            total_visits=self.total_decisions,
            source_depth=key.source_depth,
        )

    def record_attempt(
        self,
        op_name: str,
        source_key: GrammarSourceKey,
    ) -> int:
        op_stat = self.operation_stats.setdefault(op_name, UCTStat())
        source_stat = self.source_stats.setdefault(source_key, UCTStat())
        op_stat.add_visit()
        source_stat.add_visit()
        decision_id = self.total_decisions
        self.total_decisions += 1
        return decision_id

    def reward(
        self,
        op_name: str,
        source_key: GrammarSourceKey,
        credit: float,
    ) -> None:
        self.operation_stats.setdefault(op_name, UCTStat()).add_credit(credit)
        self.source_stats.setdefault(source_key, UCTStat()).add_credit(credit)

    def depth_distribution(self) -> dict[int, dict[str, list[float]]]:
        result: dict[int, dict[str, list[float]]] = {}

        for key, stat in self.source_stats.items():
            bucket = result.setdefault(
                int(key.source_depth),
                {
                    "exploitation": [],
                    "exploration": [],
                },
            )
            _, exploitation, exploration = self.source_score(key)
            bucket["exploitation"].append(exploitation)
            bucket["exploration"].append(exploration)

        return dict(sorted(result.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "exploration_start": self.exploration_start,
            "exploration_end": self.exploration_end,
            "exploration_horizon": self.exploration_horizon,
            "depth_focus": self.depth_focus,
            "total_gene_generations": self.total_gene_generations,
            "total_decisions": self.total_decisions,
            "task_attempts": self.task_attempts,
            "solved_task_ids": sorted(self.solved_task_ids),
            "operation_stats": {
                name: asdict(stat)
                for name, stat in self.operation_stats.items()
            },
            "source_stats": [
                {
                    "key": {
                        "op_name": key.op_name,
                        "source_depth": key.source_depth,
                        "source_dims": list(key.source_dims),
                        "source_dtypes": list(key.source_dtypes),
                        "source_shapes": list(key.source_shapes),
                    },
                    "stat": asdict(stat),
                }
                for key, stat in self.source_stats.items()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GrammarUCTPolicy":
        policy = cls(
            gamma=float(payload.get("gamma", 0.85)),
            exploration_start=float(
                payload.get("exploration_start", 8.0)
            ),
            exploration_end=float(payload.get("exploration_end", 0.05)),
            exploration_horizon=int(
                payload.get("exploration_horizon", 1_000_000)
            ),
            depth_focus=float(payload.get("depth_focus", 0.08)),
            total_gene_generations=int(
                payload.get("total_gene_generations", 0)
            ),
            total_decisions=int(payload.get("total_decisions", 0)),
            task_attempts=int(payload.get("task_attempts", 0)),
            solved_task_ids=set(payload.get("solved_task_ids", [])),
        )

        policy.operation_stats = {
            str(name): UCTStat(
                visits=int(stat.get("visits", 0)),
                solve_credit=float(stat.get("solve_credit", 0.0)),
            )
            for name, stat in payload.get(
                "operation_stats",
                {},
            ).items()
        }

        for item in payload.get("source_stats", []):
            key_payload = item["key"]
            stat_payload = item["stat"]
            key = GrammarSourceKey(
                op_name=str(key_payload["op_name"]),
                source_depth=int(key_payload["source_depth"]),
                source_dims=tuple(
                    int(value)
                    for value in key_payload["source_dims"]
                ),
                source_dtypes=tuple(
                    str(value)
                    for value in key_payload["source_dtypes"]
                ),
                source_shapes=tuple(
                    str(value)
                    for value in key_payload["source_shapes"]
                ),
            )
            policy.source_stats[key] = UCTStat(
                visits=int(stat_payload.get("visits", 0)),
                solve_credit=float(
                    stat_payload.get("solve_credit", 0.0)
                ),
            )

        return policy

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True)
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GrammarUCTPolicy":
        return cls.from_dict(
            json.loads(Path(path).read_text())
        )


@dataclass(frozen=True)
class _CoverageCandidate:
    op_name: str
    source_gene_ids: tuple[int, ...]


@dataclass
class _DecisionRecord:
    decision_id: int
    op_name: str
    source_key: GrammarSourceKey
    parent_gene_ids: tuple[int, ...]
    output_gene_ids: tuple[int, ...]


@dataclass
class _TaskUCTState:
    run_id: int
    seed_gene_ids: tuple[int, ...]
    gene_depth: dict[int, int]
    phenotype_cache: dict[int, tuple[int, str, str]]
    coverage_queue: deque[_CoverageCandidate]
    gene_to_decision: dict[int, int] = field(default_factory=dict)
    decisions: dict[int, _DecisionRecord] = field(default_factory=dict)
    attempted_transitions: set[tuple[str, tuple[int, ...], str]] = field(
        default_factory=set
    )

    @classmethod
    def initialize(
        cls,
        run_id: int,
        GP_meta: ProgramMeta,
        GP_X: ProgramX,
        policy: GrammarUCTPolicy,
        rng: np.random.Generator,
    ) -> "_TaskUCTState":
        seed_gene_ids = tuple(
            GP_meta.stable_id(gidx)
            for gidx in range(len(GP_meta))
        )

        state = cls(
            run_id=int(run_id),
            seed_gene_ids=seed_gene_ids,
            gene_depth={
                gene_id: 0
                for gene_id in seed_gene_ids
            },
            phenotype_cache={},
            coverage_queue=deque(),
        )
        coverage = _build_depth0_coverage(
            GP_meta,
            GP_X,
            seed_gene_ids,
        )
        coverage_probability = min(
            1.0,
            policy.exploration_coefficient(),
        )

        if coverage_probability < 1.0:
            coverage = [
                candidate
                for candidate in coverage
                if float(rng.random()) < coverage_probability
            ]

        state.coverage_queue = deque(coverage)
        return state

    def stable_to_gidx(self, GP_meta: ProgramMeta) -> dict[int, int]:
        return {
            GP_meta.stable_id(gidx): gidx
            for gidx in range(len(GP_meta))
        }

    def phenotype(
        self,
        GP_meta: ProgramMeta,
        GP_X: ProgramX,
        gidx: int,
    ) -> tuple[int, str, str]:
        gene_id = GP_meta.stable_id(gidx)

        if gene_id not in self.phenotype_cache:
            self.phenotype_cache[gene_id] = (
                int(GP_meta.dims[gidx]),
                _dtype_signature(GP_X, gidx),
                _shape_signature(GP_X[gidx]),
            )

        return self.phenotype_cache[gene_id]

    def source_key(
        self,
        GP_meta: ProgramMeta,
        GP_X: ProgramX,
        info: OperationInfo,
        source_idx: Any,
    ) -> GrammarSourceKey:
        sources = tuple(int(value) for value in _source_tuple(source_idx))

        components = []
        depths = []

        for gidx in sources:
            gene_id = GP_meta.stable_id(gidx)
            depths.append(int(self.gene_depth.get(gene_id, 0)))
            components.append(
                self.phenotype(
                    GP_meta,
                    GP_X,
                    gidx,
                )
            )

        if not info.ordered_sources:
            components = sorted(components)

        return GrammarSourceKey(
            op_name=info.name,
            source_depth=max(depths, default=0),
            source_dims=tuple(item[0] for item in components),
            source_dtypes=tuple(item[1] for item in components),
            source_shapes=tuple(item[2] for item in components),
        )

    def transition_token(
        self,
        GP_meta: ProgramMeta,
        info: OperationInfo,
        source_idx: Any,
        params: dict[str, Any],
    ) -> tuple[str, tuple[int, ...], str]:
        source_gene_ids = tuple(
            GP_meta.stable_id(int(gidx))
            for gidx in _source_tuple(source_idx)
        )

        if not info.ordered_sources:
            source_gene_ids = tuple(sorted(source_gene_ids))

        params_token = json.dumps(
            params,
            sort_keys=True,
            default=repr,
        )
        return (
            info.name,
            source_gene_ids,
            params_token,
        )

    def has_attempted(
        self,
        GP_meta: ProgramMeta,
        info: OperationInfo,
        source_idx: Any,
        params: dict[str, Any],
    ) -> bool:
        return (
            self.transition_token(
                GP_meta,
                info,
                source_idx,
                params,
            )
            in self.attempted_transitions
        )

    def mark_attempted(
        self,
        GP_meta: ProgramMeta,
        info: OperationInfo,
        source_idx: Any,
        params: dict[str, Any],
    ) -> None:
        self.attempted_transitions.add(
            self.transition_token(
                GP_meta,
                info,
                source_idx,
                params,
            )
        )

    def register_generation(
        self,
        policy: GrammarUCTPolicy,
        GP_meta: ProgramMeta,
        decision_id: int,
        info: OperationInfo,
        source_idx: Any,
        source_key: GrammarSourceKey,
        generated_gidxs: list[int],
    ) -> None:
        parent_gidxs = tuple(
            int(value)
            for value in _source_tuple(source_idx)
        )
        parent_gene_ids = tuple(
            GP_meta.stable_id(gidx)
            for gidx in parent_gidxs
        )
        source_depth = max(
            (
                self.gene_depth.get(gene_id, 0)
                for gene_id in parent_gene_ids
            ),
            default=0,
        )
        output_gene_ids = tuple(
            GP_meta.stable_id(gidx)
            for gidx in generated_gidxs
        )

        record = _DecisionRecord(
            decision_id=decision_id,
            op_name=info.name,
            source_key=source_key,
            parent_gene_ids=parent_gene_ids,
            output_gene_ids=output_gene_ids,
        )
        self.decisions[decision_id] = record

        for gene_id in output_gene_ids:
            self.gene_depth[gene_id] = source_depth + 1
            self.gene_to_decision[gene_id] = decision_id

        policy.total_gene_generations += len(output_gene_ids)

    def reward_solver_genes(
        self,
        policy: GrammarUCTPolicy,
        solver_gene_ids: Iterable[int],
    ) -> None:
        decision_credit: dict[int, float] = {}
        queue: deque[tuple[int, float]] = deque(
            (int(gene_id), 1.0)
            for gene_id in set(solver_gene_ids)
        )

        while queue:
            gene_id, credit = queue.popleft()
            decision_id = self.gene_to_decision.get(gene_id)

            if decision_id is None:
                continue

            prior = decision_credit.get(decision_id, 0.0)
            if credit <= prior:
                continue

            decision_credit[decision_id] = credit
            record = self.decisions[decision_id]

            parent_credit = credit * policy.gamma

            if parent_credit <= 0.0:
                continue

            for parent_gene_id in record.parent_gene_ids:
                queue.append(
                    (parent_gene_id, parent_credit)
                )

        for decision_id, credit in decision_credit.items():
            record = self.decisions[decision_id]
            policy.reward(
                record.op_name,
                record.source_key,
                credit,
            )


@dataclass
class CrawlTaskAttempt:
    task_id: str
    solved_exactly: bool
    generated_genes: int
    iterations: int
    final_gp_len: int
    final_sp_len: int
    elapsed_seconds: float


@dataclass
class CrawlResult:
    policy: GrammarUCTPolicy
    attempts: list[CrawlTaskAttempt]
    solved_task_ids: set[str]
    solutions: dict[str, Kelschinator]


def _dtype_signature(X: ProgramX, gidx: int) -> str:
    return ",".join(
        str(dtype)
        for dtype in gene_atomic_dtypes(X, gidx)
    )


def _shape_signature(gene: np.ndarray) -> str:
    shapes = [
        tuple(np.asarray(value).shape)
        for value in gene
    ]

    if not shapes:
        return "empty"

    ndim = len(shapes[0])

    if ndim == 0:
        return "scalar"

    fixed = len(set(shapes)) == 1
    sizes = [
        int(np.prod(shape))
        for shape in shapes
    ]
    median_size = float(np.median(np.asarray(sizes, dtype=float)))

    if median_size <= 9:
        size_bucket = "small"
    elif median_size <= 36:
        size_bucket = "medium"
    else:
        size_bucket = "large"

    if ndim == 2:
        square = all(
            len(shape) == 2 and shape[0] == shape[1]
            for shape in shapes
        )
        geometry = "square" if square else "rect"
    else:
        geometry = f"{ndim}d"

    variability = "fixed" if fixed else "variable"

    return f"{geometry}:{variability}:{size_bucket}"


def _build_depth0_coverage(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    seed_gene_ids: tuple[int, ...],
) -> list[_CoverageCandidate]:
    """Build one deterministic legal-shape coverage ordering over seed GP."""

    seed_gidxs = list(range(len(seed_gene_ids)))
    op_infos = _eligible_operation_infos(side="GP")
    op_order = {
        info.name: position
        for position, info in enumerate(op_infos)
    }

    static_allowed = {
        info.name: set(_static_source_indices(GP_meta, GP_X, info))
        for info in op_infos
    }

    candidates: list[
        tuple[
            tuple[int, int, tuple[int, ...], int],
            _CoverageCandidate,
        ]
    ] = []

    for info in op_infos:
        allowed = static_allowed[info.name]

        if info.source_count == 1:
            source_iterable = (
                (gidx,)
                for gidx in seed_gidxs
                if gidx in allowed
            )
        elif info.ordered_sources:
            source_iterable = (
                tuple(source)
                for source in permutations(
                    seed_gidxs,
                    info.source_count,
                )
                if all(gidx in allowed for gidx in source)
            )
        else:
            source_iterable = (
                tuple(source)
                for source in combinations(
                    seed_gidxs,
                    info.source_count,
                )
                if all(gidx in allowed for gidx in source)
            )

        for source in source_iterable:
            gene_ids = tuple(
                seed_gene_ids[gidx]
                for gidx in source
            )
            sort_key = (
                max(source),
                len(source),
                tuple(source),
                op_order[info.name],
            )
            candidates.append(
                (
                    sort_key,
                    _CoverageCandidate(
                        op_name=info.name,
                        source_gene_ids=gene_ids,
                    ),
                )
            )

    candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in candidates]


def _solver_gene_ids(
    ST: SolutionTree,
    GP_meta: ProgramMeta,
) -> set[int]:
    result: set[int] = set()

    for node in ST.nodes.values():
        gp_gidx = getattr(node, "gp_gidx", -1)

        if (
            isinstance(gp_gidx, (int, np.integer))
            and 0 <= int(gp_gidx) < len(GP_meta)
        ):
            result.add(
                GP_meta.stable_id(int(gp_gidx))
            )

    return result


def _choose_operation(
    policy: GrammarUCTPolicy,
    infos: list[OperationInfo],
) -> OperationInfo:
    ranked = sorted(
        enumerate(infos),
        key=lambda item: (
            -policy.operation_score(item[1].name)[0],
            item[0],
        ),
    )
    return ranked[0][1]


def _source_component_scores(
    policy: GrammarUCTPolicy,
    task_state: _TaskUCTState,
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    info: OperationInfo,
    candidate_gidxs: list[int],
) -> np.ndarray:
    scores = []

    for gidx in candidate_gidxs:
        key = task_state.source_key(
            GP_meta,
            GP_X,
            info,
            gidx,
        )
        scores.append(policy.source_score(key)[0])

    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores

    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _select_source_for_operation(
    policy: GrammarUCTPolicy,
    task_state: _TaskUCTState,
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    info: OperationInfo,
    params: dict[str, Any],
    *,
    rng: np.random.Generator,
    source_probe_attempts: int,
) -> tuple[Any, GrammarSourceKey] | None:
    candidates = _static_source_indices(
        GP_meta,
        GP_X,
        info,
    )

    if len(candidates) < info.source_count:
        return None

    if info.source_count == 1:
        ranked: list[
            tuple[float, int, int, GrammarSourceKey]
        ] = []

        for gidx in candidates:
            key = task_state.source_key(
                GP_meta,
                GP_X,
                info,
                gidx,
            )
            score = policy.source_score(key)[0]
            gene_id = GP_meta.stable_id(gidx)
            depth = task_state.gene_depth.get(gene_id, 0)
            ranked.append(
                (
                    score,
                    depth,
                    gidx,
                    key,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

        for _, _, gidx, key in ranked:
            if task_state.has_attempted(
                GP_meta,
                info,
                gidx,
                params,
            ):
                continue

            if valid_generation(
                GP_meta,
                GP_X,
                info,
                gidx,
                params=params,
            ):
                task_state.mark_attempted(
                    GP_meta,
                    info,
                    gidx,
                    params,
                )
                return gidx, key

            task_state.mark_attempted(
                GP_meta,
                info,
                gidx,
                params,
            )

        return None

    probabilities = _source_component_scores(
        policy,
        task_state,
        GP_meta,
        GP_X,
        info,
        candidates,
    )
    best: tuple[float, Any, GrammarSourceKey] | None = None

    for probe in range(max(1, int(source_probe_attempts))):
        if probe == 0:
            ranked_positions = np.argsort(-probabilities)
            chosen_positions = ranked_positions[: info.source_count]
        else:
            chosen_positions = rng.choice(
                len(candidates),
                size=info.source_count,
                replace=False,
                p=probabilities,
            )

        source = tuple(
            candidates[int(position)]
            for position in chosen_positions
        )

        if not info.ordered_sources:
            source = tuple(sorted(source))

        if task_state.has_attempted(
            GP_meta,
            info,
            source,
            params,
        ):
            continue

        if not valid_generation(
            GP_meta,
            GP_X,
            info,
            source,
            params=params,
        ):
            task_state.mark_attempted(
                GP_meta,
                info,
                source,
                params,
            )
            continue

        key = task_state.source_key(
            GP_meta,
            GP_X,
            info,
            source,
        )
        score = policy.source_score(key)[0]

        if best is None or score > best[0]:
            best = (score, source, key)

    if best is None:
        return None

    task_state.mark_attempted(
        GP_meta,
        info,
        best[1],
        params,
    )
    return best[1], best[2]


def _next_uct_candidate(
    policy: GrammarUCTPolicy,
    task_state: _TaskUCTState,
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    *,
    rng: np.random.Generator,
    source_probe_attempts: int,
) -> tuple[OperationInfo, Any, dict[str, Any], GrammarSourceKey] | None:
    """Return depth-0 coverage first, then operation->source UCT choices."""

    stable_to_gidx = task_state.stable_to_gidx(GP_meta)

    while task_state.coverage_queue:
        candidate = task_state.coverage_queue.popleft()

        if any(
            gene_id not in stable_to_gidx
            for gene_id in candidate.source_gene_ids
        ):
            continue

        info = OP_REGISTRY.get(candidate.op_name)
        if info is None:
            continue

        source_tuple = tuple(
            stable_to_gidx[gene_id]
            for gene_id in candidate.source_gene_ids
        )
        source_idx: Any = (
            source_tuple[0]
            if len(source_tuple) == 1
            else source_tuple
        )
        params = sample_operation_params(info, rng)

        if task_state.has_attempted(
            GP_meta,
            info,
            source_idx,
            params,
        ):
            continue

        if not valid_generation(
            GP_meta,
            GP_X,
            info,
            source_idx,
            params=params,
        ):
            task_state.mark_attempted(
                GP_meta,
                info,
                source_idx,
                params,
            )
            continue

        task_state.mark_attempted(
            GP_meta,
            info,
            source_idx,
            params,
        )
        key = task_state.source_key(
            GP_meta,
            GP_X,
            info,
            source_idx,
        )
        return info, source_idx, params, key

    remaining = _eligible_operation_infos(side="GP")

    while remaining:
        info = _choose_operation(policy, remaining)
        params = sample_operation_params(info, rng)
        selected = _select_source_for_operation(
            policy,
            task_state,
            GP_meta,
            GP_X,
            info,
            params,
            rng=rng,
            source_probe_attempts=source_probe_attempts,
        )

        if selected is not None:
            source_idx, source_key = selected
            return info, source_idx, params, source_key

        remaining = [
            candidate
            for candidate in remaining
            if candidate.name != info.name
        ]

    return None


def _training_task_ids(data_root: str | Path | None) -> list[str]:
    root = (
        _repo_data_root()
        if data_root is None
        else Path(data_root)
    )
    training_dir = root / "data" / "training"

    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: {training_dir}"
        )

    task_ids = sorted(
        path.stem
        for path in training_dir.glob("*.json")
    )

    if not task_ids:
        raise FileNotFoundError(
            f"No training tasks found in {training_dir}"
        )

    return task_ids


def _plot_depth_distribution(
    policy: GrammarUCTPolicy,
    *,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    distribution = policy.depth_distribution()

    if not distribution:
        return

    depths = list(distribution)
    exploitation = [
        distribution[depth]["exploitation"]
        for depth in depths
    ]
    exploration = [
        distribution[depth]["exploration"]
        for depth in depths
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].boxplot(
        exploitation,
        tick_labels=[str(depth) for depth in depths],
        showfliers=False,
    )
    axes[0].set_title("Exploitation by source depth")
    axes[0].set_xlabel("Source depth")
    axes[0].set_ylabel("sqrt(solve credit / visits)")

    axes[1].boxplot(
        exploration,
        tick_labels=[str(depth) for depth in depths],
        showfliers=False,
    )
    axes[1].set_title("Exploration bonus by source depth")
    axes[1].set_xlabel("Source depth")
    axes[1].set_ylabel("UCT exploration term")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _plot_success(
    task_id: str,
    task: Any,
    kelschinator: Kelschinator,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    expected_outputs = task.test_outputs

    for index, test_input in enumerate(task.test_inputs):
        prediction = kelschinator.transform(
            np.asarray(test_input, dtype=np.int64)
        )
        expected = (
            None
            if expected_outputs is None
            else np.asarray(
                expected_outputs[index],
                dtype=np.int64,
            )
        )

        column_count = 2 if expected is None else 3
        fig, axes = plt.subplots(1, column_count, figsize=(4 * column_count, 4))

        if column_count == 2:
            axes = list(axes)
        else:
            axes = list(axes)

        panels = [
            ("Test input", np.asarray(test_input, dtype=np.int64)),
            ("Predicted output", prediction),
        ]

        if expected is not None:
            panels.append(("Expected output", expected))

        for axis, (panel_title, grid) in zip(axes, panels):
            axis.imshow(
                grid,
                interpolation="nearest",
                vmin=0,
                vmax=9,
            )
            axis.set_title(panel_title)
            axis.set_xticks([])
            axis.set_yticks([])

        fig.suptitle(f"{task_id} | test {index}")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def _run_crawl_task(
    task_id: str,
    policy: GrammarUCTPolicy,
    *,
    run_id: int,
    max_GP: int,
    max_SP: int,
    prune_size_GP: int,
    task_generation_budget: int,
    global_generation_budget: int,
    source_probe_attempts: int,
    rng: np.random.Generator,
    data_root: str | Path | None,
    select_residual_max_destinations: int,
    verbosity: int,
    status_every: int,
    plot_every_generations: int,
) -> tuple[CrawlTaskAttempt, Kelschinator | None]:
    started = perf_counter()
    task = _load_task_by_id(
        task_id,
        data_root=data_root,
        split="training",
    )

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(
        task.train,
        select_residual_max_destinations=(
            select_residual_max_destinations
        ),
    )
    task_state = _TaskUCTState.initialize(
        run_id,
        GP_meta,
        GP_X,
        policy,
        rng,
    )
    evaluation_cache = SolveEvaluationCache()
    generated_this_task = 0
    iterations = 0
    gp_exhausted = False
    sp_exhausted = False
    stalled_iterations = 0

    before_solver_ids = _solver_gene_ids(ST, GP_meta)
    _solve_frontier(
        GP_meta,
        GP_X,
        SP_meta,
        SP_X,
        ST,
        evaluation_cache,
    )
    initial_new_solvers = (
        _solver_gene_ids(ST, GP_meta)
        - before_solver_ids
    )
    task_state.reward_solver_genes(
        policy,
        initial_new_solvers,
    )

    while (
        not ST.solved
        and generated_this_task < task_generation_budget
        and policy.total_gene_generations < global_generation_budget
    ):
        iterations += 1
        iteration_progress = False

        if len(GP_X) >= max_GP:
            if prune_size_GP > 0:
                removed = GP_prune(
                    GP_meta,
                    GP_X,
                    ST,
                    prune=prune_size_GP,
                    rng=rng,
                )
                if removed:
                    iteration_progress = True
                else:
                    gp_exhausted = True
            else:
                gp_exhausted = True

        generated_gp: list[int] = []

        if not gp_exhausted and len(GP_X) < max_GP:
            candidate = _next_uct_candidate(
                policy,
                task_state,
                GP_meta,
                GP_X,
                rng=rng,
                source_probe_attempts=source_probe_attempts,
            )

            if candidate is None:
                gp_exhausted = True
            else:
                info, source_idx, params, source_key = candidate
                decision_id = policy.record_attempt(
                    info.name,
                    source_key,
                )
                generated_gp, _ = _try_candidate_transactionally(
                    GP_meta,
                    GP_X,
                    info,
                    source_idx,
                    params,
                )

                if generated_gp:
                    task_state.register_generation(
                        policy,
                        GP_meta,
                        decision_id,
                        info,
                        source_idx,
                        source_key,
                        generated_gp,
                    )
                    generated_count = len(generated_gp)
                    generated_this_task += generated_count
                    iteration_progress = True

        if (
            not sp_exhausted
            and not ST.solved
            and len(SP_X) < max_SP
        ):
            generated_sp = SP_generate(
                SP_meta,
                SP_X,
                ST,
                rng=rng,
            )

            if generated_sp:
                iteration_progress = True
            else:
                sp_exhausted = True
        elif len(SP_X) >= max_SP:
            sp_exhausted = True

        before_solver_ids = _solver_gene_ids(ST, GP_meta)
        _solve_frontier(
            GP_meta,
            GP_X,
            SP_meta,
            SP_X,
            ST,
            evaluation_cache,
        )
        new_solver_ids = (
            _solver_gene_ids(ST, GP_meta)
            - before_solver_ids
        )

        if new_solver_ids:
            iteration_progress = True
            task_state.reward_solver_genes(
                policy,
                new_solver_ids,
            )

        if len(GP_X) > max_GP and prune_size_GP > 0 and not ST.solved:
            GP_prune(
                GP_meta,
                GP_X,
                ST,
                prune=prune_size_GP,
                rng=rng,
            )

        if iteration_progress:
            stalled_iterations = 0
        else:
            stalled_iterations += 1

        if verbosity >= 2 and (
            iterations == 1
            or iterations % max(1, status_every) == 0
            or ST.solved
        ):
            solved_nodes, total_nodes, proportion = _st_progress(ST)
            print(
                f"  iter={iterations} "
                f"task_genes={generated_this_task} "
                f"total_genes={policy.total_gene_generations} "
                f"C={policy.exploration_coefficient():.4f} "
                f"ST={solved_nodes}/{total_nodes} "
                f"({proportion:.1%}) "
                f"GP={len(GP_X)} SP={len(SP_X)}"
            )

        if (
            verbosity >= 3
            and plot_every_generations > 0
            and generated_this_task > 0
            and generated_this_task % plot_every_generations == 0
        ):
            _plot_depth_distribution(
                policy,
                title=(
                    f"{task_id} | total GP generations "
                    f"{policy.total_gene_generations:,}"
                ),
            )

        if stalled_iterations >= 25:
            break

        if gp_exhausted and sp_exhausted and not iteration_progress:
            break

    kelschinator = Kelschinator()
    solved_exactly = False

    if ST.solved and kelschinator.fit(ST):
        solved_exactly = _exact_test_match(
            task,
            kelschinator,
        )

    elapsed = perf_counter() - started
    attempt = CrawlTaskAttempt(
        task_id=task_id,
        solved_exactly=bool(solved_exactly),
        generated_genes=int(generated_this_task),
        iterations=int(iterations),
        final_gp_len=len(GP_X),
        final_sp_len=len(SP_X),
        elapsed_seconds=float(elapsed),
    )

    if solved_exactly:
        return attempt, kelschinator

    return attempt, None


def crawl_synth_v1(
    first_tasks: list[str] | None = None,
    *,
    max_GP: int = 1000,
    max_SP: int = 100,
    prune_size_GP: int = 5,
    max_total_generations: int = 1_000_000,
    max_task_generations: int | None = None,
    gamma: float = 0.85,
    exploration_start: float = 8.0,
    exploration_end: float = 0.05,
    exploration_horizon: int = 1_000_000,
    depth_focus: float = 0.08,
    source_probe_attempts: int = 32,
    rng: np.random.Generator | int | None = None,
    data_root: str | Path | None = None,
    select_residual_max_destinations: int = 5,
    verbosity: int = 1,
    status_every: int = 100,
    plot_every_generations: int = 500,
    state_path: str | Path | None = None,
    show_success_plots: bool = True,
    stall_task_limit: int = 50,
) -> CrawlResult:
    """Crawl ARC training tasks with a persistent cross-task Grammar-UCT policy.

    Task order:
      1. consume first_tasks in the supplied order;
      2. then sample with replacement from the currently unsolved training set;
      3. remove a task from that sampling set only after exact test success.

    Every task may begin with low-index depth-0 coverage over the fixed
    initialization GP pool. Coverage probability is min(1, C), where C is the
    decaying global exploration coefficient: it is exhaustive at the start and
    falls to about 5% when C reaches 0.05. After that, GP generation uses
    operation-first UCT followed by source-phenotype UCT.

    Exploitation is sqrt(solve_credit / visits). A newly successful GP solver
    sends reward backward through generated GP lineage with discount gamma.

    The exploration coefficient decays exponentially from exploration_start to
    exploration_end over exploration_horizon accepted GP gene generations.
    """
    for name, value in (
        ("max_GP", max_GP),
        ("max_SP", max_SP),
        ("prune_size_GP", prune_size_GP),
        ("max_total_generations", max_total_generations),
        ("source_probe_attempts", source_probe_attempts),
        ("status_every", status_every),
        ("stall_task_limit", stall_task_limit),
    ):
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    max_GP = int(max_GP)
    max_SP = int(max_SP)
    prune_size_GP = int(prune_size_GP)
    max_total_generations = int(max_total_generations)
    source_probe_attempts = int(source_probe_attempts)
    status_every = int(status_every)
    stall_task_limit = int(stall_task_limit)

    if max_GP < 1:
        raise ValueError("max_GP must be >= 1.")
    if max_SP < 1:
        raise ValueError("max_SP must be >= 1.")
    if source_probe_attempts < 1:
        raise ValueError("source_probe_attempts must be >= 1.")
    if stall_task_limit < 1:
        raise ValueError("stall_task_limit must be >= 1.")
    if isinstance(verbosity, bool) or verbosity not in {0, 1, 2, 3}:
        raise ValueError("verbosity must be one of 0, 1, 2, or 3.")

    if max_task_generations is None:
        max_task_generations = max_GP
    if (
        isinstance(max_task_generations, bool)
        or int(max_task_generations) < 1
    ):
        raise ValueError("max_task_generations must be a positive integer.")
    max_task_generations = int(max_task_generations)

    rng = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )

    state_file = None if state_path is None else Path(state_path)

    if state_file is not None and state_file.exists():
        policy = GrammarUCTPolicy.load(state_file)
        policy.gamma = float(gamma)
        policy.exploration_start = float(exploration_start)
        policy.exploration_end = float(exploration_end)
        policy.exploration_horizon = int(exploration_horizon)
        policy.depth_focus = float(depth_focus)
        policy.__post_init__()
    else:
        policy = GrammarUCTPolicy(
            gamma=gamma,
            exploration_start=exploration_start,
            exploration_end=exploration_end,
            exploration_horizon=exploration_horizon,
            depth_focus=depth_focus,
        )

    training_task_ids = _training_task_ids(data_root)
    training_set = set(training_task_ids)
    first_queue = deque(
        str(task_id)
        for task_id in (first_tasks or [])
    )

    unknown_first = [
        task_id
        for task_id in first_queue
        if task_id not in training_set
    ]

    if unknown_first:
        raise ValueError(
            "first_tasks contains task IDs not present in the training set: "
            + ", ".join(unknown_first)
        )

    policy.solved_task_ids.intersection_update(training_set)
    unsolved = training_set - policy.solved_task_ids
    attempts: list[CrawlTaskAttempt] = []
    solutions: dict[str, Kelschinator] = {}
    zero_progress_task_attempts = 0

    while (
        unsolved
        and policy.total_gene_generations < max_total_generations
    ):
        task_id = None

        while first_queue and task_id is None:
            candidate = first_queue.popleft()
            if candidate in unsolved:
                task_id = candidate

        if task_id is None:
            choices = sorted(unsolved)
            task_id = choices[
                int(rng.integers(len(choices)))
            ]

        policy.task_attempts += 1
        run_id = policy.task_attempts

        if verbosity >= 1:
            print(
                f"[crawl task {policy.task_attempts}] {task_id} | "
                f"solved={len(policy.solved_task_ids)}/{len(training_set)} | "
                f"GP generations={policy.total_gene_generations:,} | "
                f"C={policy.exploration_coefficient():.4f}"
            )

        remaining_global = (
            max_total_generations
            - policy.total_gene_generations
        )
        task_budget = min(
            max_task_generations,
            max(1, remaining_global),
        )

        attempt, solution = _run_crawl_task(
            task_id,
            policy,
            run_id=run_id,
            max_GP=max_GP,
            max_SP=max_SP,
            prune_size_GP=prune_size_GP,
            task_generation_budget=task_budget,
            global_generation_budget=max_total_generations,
            source_probe_attempts=source_probe_attempts,
            rng=rng,
            data_root=data_root,
            select_residual_max_destinations=(
                select_residual_max_destinations
            ),
            verbosity=verbosity,
            status_every=status_every,
            plot_every_generations=plot_every_generations,
        )
        attempts.append(attempt)

        if attempt.generated_genes == 0:
            zero_progress_task_attempts += 1
        else:
            zero_progress_task_attempts = 0

        if attempt.solved_exactly and solution is not None:
            policy.solved_task_ids.add(task_id)
            unsolved.discard(task_id)
            solutions[task_id] = solution

            if verbosity >= 1:
                print(
                    f"  SOLVED {task_id} in {attempt.elapsed_seconds:.2f}s "
                    f"with {attempt.generated_genes} generated GP genes."
                )

            if show_success_plots:
                task = _load_task_by_id(
                    task_id,
                    data_root=data_root,
                    split="training",
                )
                _plot_success(
                    task_id,
                    task,
                    solution,
                )
        elif verbosity >= 1:
            print(
                f"  unresolved after {attempt.generated_genes} GP genes "
                f"({attempt.elapsed_seconds:.2f}s)"
            )

        if verbosity >= 3:
            _plot_depth_distribution(
                policy,
                title=(
                    f"Grammar-UCT after {policy.total_gene_generations:,} "
                    "GP generations"
                ),
            )

        if state_file is not None:
            policy.save(state_file)

        if zero_progress_task_attempts >= stall_task_limit:
            if verbosity >= 1:
                print(
                    "Stopping crawl after repeated zero-generation task "
                    "attempts."
                )
            break

    if state_file is not None:
        policy.save(state_file)

    return CrawlResult(
        policy=policy,
        attempts=attempts,
        solved_task_ids=set(policy.solved_task_ids),
        solutions=solutions,
    )
