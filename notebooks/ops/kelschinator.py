from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .environment import (
    ProgramMeta,
    ProgramX,
    STInverseRef,
    STNodeId,
    STNodeRef,
    STSet,
    SolutionTree,
    source_indices,
)
from .inv_ops import INV_OP_REGISTRY
from .ops import OP_REGISTRY


class KelschinatorCompileError(RuntimeError):
    """Raised internally when a solved ST proof cannot be executed."""


@dataclass(frozen=True)
class _GPInstruction:
    gidx: int
    op: str
    sources: tuple[int, ...]
    params: dict[str, Any]
    output_slot: int
    aux: dict[str, Any]


@dataclass(frozen=True)
class _DirectPlan:
    node_id: STNodeId
    gp_gidx: int
    rule: str
    params: dict[str, Any]
    target_dtype: np.dtype
    dims: int


@dataclass(frozen=True)
class _InversePlan:
    inverse_op: str
    children: tuple[Any, ...]


@dataclass(frozen=True)
class _BundlePlan:
    children: tuple[Any, ...]


@dataclass(frozen=True)
class _BundleValue:
    values: tuple[Any, ...]


def _copy_param(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {
            str(key): _copy_param(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_param(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_param(item) for item in value)
    return value


def _freeze_value(value: Any):
    if isinstance(value, np.generic):
        return ("numpy_scalar", str(value.dtype), value.item())

    if isinstance(value, np.ndarray):
        return (
            "array",
            str(value.dtype),
            tuple(value.shape),
            tuple(_freeze_value(item) for item in value.flat),
        )

    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        str(key),
                        _freeze_value(item),
                    )
                    for key, item in value.items()
                )
            ),
        )

    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_freeze_value(item) for item in value),
        )

    return value


def _meta_signature(
    meta: ProgramMeta,
    gidx: int,
):
    return (
        meta.op[gidx],
        _freeze_value(meta.source[gidx]),
        _freeze_value(meta.params[gidx]),
    )


def _value_dims(value: Any) -> int:
    return int(np.asarray(value).ndim)


def _structural_offset(
    position: str,
    container_shape: tuple[int, int],
    item_shape: tuple[int, int],
) -> tuple[int, int]:
    ch, cw = container_shape
    ih, iw = item_shape

    if ih > ch or iw > cw:
        raise ValueError(
            "Item does not fit inside requested container."
        )

    if position == "top_left":
        return (0, 0)
    if position == "top_right":
        return (0, cw - iw)
    if position == "bottom_left":
        return (ch - ih, 0)
    if position == "bottom_right":
        return (ch - ih, cw - iw)
    if position == "center":
        dh = ch - ih
        dw = cw - iw
        if dh % 2 != 0 or dw % 2 != 0:
            raise ValueError(
                "Center placement is ambiguous for this shape pair."
            )
        return (dh // 2, dw // 2)

    raise ValueError(f"Unknown structural position {position!r}.")


def _shift_zero(
    source: np.ndarray,
    dr: int,
    dc: int,
    dtype: np.dtype,
) -> np.ndarray:
    source = np.asarray(source, dtype=dtype)
    h, w = source.shape
    result = np.zeros(source.shape, dtype=dtype)

    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return result

    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc

    result[
        dst_r0:dst_r1,
        dst_c0:dst_c1,
    ] = source[
        src_r0:src_r1,
        src_c0:src_c1,
    ]

    return result


def _exact_equal(a: Any, b: Any) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)

    if aa.shape != bb.shape or aa.dtype != bb.dtype:
        return False

    try:
        return bool(np.array_equal(aa, bb, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(aa, bb))


class Kelschinator:
    """Distill a solved SolutionTree into an executable input->output program.

    Typical use:

        kelschinator = Kelschinator()
        fit_success = kelschinator.fit(ST)

        if fit_success:
            y_hat = kelschinator.transform(X_test)

    fit() returns False when:
      - ST is not fully solved;
      - the ST is not bound to its GP/SP environment;
      - the satisfied Boolean proof contains a non-reconstructive step;
      - a learned spatial rule is not executable on unseen input;
      - the distilled program fails to exactly reproduce any training output.

    After a successful fit, only the distilled proof and the GP dependencies
    required by that proof are retained.
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self.last_error_: str | None = None
        self.pipeline_: tuple[str, ...] = ()
        self._root_plan: Any | None = None
        self._gp_instructions: dict[int, _GPInstruction] = {}

    def _reset(self) -> None:
        self.is_fitted_ = False
        self.last_error_ = None
        self.pipeline_ = ()
        self._root_plan = None
        self._gp_instructions = {}

    def fit(self, ST: SolutionTree) -> bool:
        """Distill a fully solved ST into an executable pipeline."""
        self._reset()

        if not isinstance(ST, SolutionTree):
            self.last_error_ = "fit requires a SolutionTree."
            return False

        if not ST.solved:
            self.last_error_ = "SolutionTree is not fully solved."
            return False

        if not ST.has_bound_environment:
            self.last_error_ = (
                "SolutionTree has no bound GP/SP environment. "
                "Use the ST returned by init_env or call ST.bind_environment()."
            )
            return False

        if len(ST.roots) != 1:
            self.last_error_ = (
                "Kelschinator currently requires exactly one ST root."
            )
            return False

        GP_meta = ST._GP_meta
        GP_X = ST._GP_X
        SP_X = ST._SP_X

        if GP_meta is None or GP_X is None or SP_X is None:
            self.last_error_ = "Bound ST environment is incomplete."
            return False

        try:
            root_plan = self._compile_node(
                ST,
                ST.roots[0],
                GP_X,
                SP_X,
            )

            required_gp = sorted(
                self._collect_direct_gp_indices(root_plan)
            )

            needed: set[int] = set()
            for gidx in required_gp:
                self._collect_gp_dependencies(
                    GP_meta,
                    gidx,
                    needed,
                )

            instructions = {
                gidx: self._compile_gp_instruction(
                    GP_meta,
                    GP_X,
                    gidx,
                )
                for gidx in sorted(needed)
            }

            # Validate the frozen program against every training pair.
            for sample_idx in range(GP_X.sample_count):
                memo: dict[int, Any] = {}
                x_train = np.asarray(
                    GP_X[0, sample_idx]
                ).copy()

                y_hat = self._eval_plan(
                    root_plan,
                    x_train,
                    instructions,
                    memo,
                    {},
                )
                y_true = np.asarray(
                    SP_X[0, sample_idx]
                )

                if not _exact_equal(y_hat, y_true):
                    raise KelschinatorCompileError(
                        "Distilled pipeline failed exact replay on "
                        f"training sample {sample_idx}."
                    )

            self._root_plan = root_plan
            self._gp_instructions = instructions
            self.pipeline_ = self._describe_pipeline(
                root_plan,
                instructions,
            )
            self.is_fitted_ = True
            return True

        except Exception as exc:
            self._reset()
            self.last_error_ = (
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def transform(self, X: Any) -> np.ndarray:
        """Apply the distilled program to one unseen input matrix."""
        if not self.is_fitted_ or self._root_plan is None:
            raise RuntimeError(
                "Kelschinator must be successfully fit before transform()."
            )

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "Kelschinator.transform currently expects one 2D input matrix."
            )

        result = self._eval_plan(
            self._root_plan,
            X.copy(),
            self._gp_instructions,
            {},
            {},
        )

        result = np.asarray(result)

        if result.ndim != 2:
            raise RuntimeError(
                "Distilled ST root did not reconstruct a 2D output matrix."
            )

        return result.copy()

    def _compile_node(
        self,
        ST: SolutionTree,
        node_id: STNodeId,
        GP_X: ProgramX,
        SP_X: ProgramX,
    ):
        node = ST[node_id]

        if node.directly_solved:
            if node.gp_gidx < 0:
                raise KelschinatorCompileError(
                    f"Node {node_id!r} is innately solved but has no value."
                )
            if node.sp_gidx is None:
                raise KelschinatorCompileError(
                    f"Directly solved node {node_id!r} has no SP gene."
                )

            rule = node.solution_rule or "identity"
            params = {
                str(key): _copy_param(value)
                for key, value in node.solution_params.items()
            }

            target_gene = SP_X[node.sp_gidx]
            target_dtype = self._gene_dtype(
                target_gene,
                expected_dims=node.dims,
            )

            if node.dims == 2:
                params = self._distill_2d_geometry(
                    rule,
                    params,
                    GP_X[node.gp_gidx],
                    target_gene,
                    root_gene=SP_X[0],
                    is_root=(node.sp_gidx == 0),
                )

            return _DirectPlan(
                node_id=node_id,
                gp_gidx=int(node.gp_gidx),
                rule=rule,
                params=params,
                target_dtype=target_dtype,
                dims=int(node.dims),
            )

        if node.derivation is None:
            raise KelschinatorCompileError(
                f"Node {node_id!r} is not executable."
            )

        return self._compile_requirement(
            ST,
            node.derivation,
            GP_X,
            SP_X,
        )

    def _compile_requirement(
        self,
        ST: SolutionTree,
        requirement: Any,
        GP_X: ProgramX,
        SP_X: ProgramX,
    ):
        if isinstance(requirement, STNodeRef):
            return self._compile_node(
                ST,
                requirement.node_id,
                GP_X,
                SP_X,
            )

        if isinstance(requirement, STInverseRef):
            raise KelschinatorCompileError(
                "Inverse operation cannot stand alone as a value."
            )

        if not isinstance(requirement, STSet):
            raise KelschinatorCompileError(
                f"Unsupported ST requirement {type(requirement).__name__}."
            )

        if requirement.mode == "OR":
            for member in requirement.members:
                if ST._eval_requirement(member):
                    return self._compile_requirement(
                        ST,
                        member,
                        GP_X,
                        SP_X,
                    )

            raise KelschinatorCompileError(
                "Solved OR set contains no satisfied branch."
            )

        inverse_refs = [
            member
            for member in requirement.members
            if isinstance(member, STInverseRef)
        ]
        value_members = [
            member
            for member in requirement.members
            if not isinstance(member, STInverseRef)
        ]

        children = tuple(
            self._compile_requirement(
                ST,
                member,
                GP_X,
                SP_X,
            )
            for member in value_members
        )

        if not inverse_refs:
            return _BundlePlan(children=children)

        if len(inverse_refs) != 1:
            raise KelschinatorCompileError(
                "Executable AND branch must contain at most one inverse."
            )

        inverse_op = inverse_refs[0].inverse_op

        try:
            inverse_info = INV_OP_REGISTRY[inverse_op]
        except KeyError as exc:
            raise KelschinatorCompileError(
                f"Missing inverse operation {inverse_op!r}."
            ) from exc

        if not inverse_info.reconstructive:
            raise KelschinatorCompileError(
                f"Inverse {inverse_op!r} is non-reconstructive."
            )

        return _InversePlan(
            inverse_op=inverse_op,
            children=children,
        )

    def _collect_direct_gp_indices(self, plan: Any) -> set[int]:
        if isinstance(plan, _DirectPlan):
            return {plan.gp_gidx}

        if isinstance(plan, (_InversePlan, _BundlePlan)):
            result: set[int] = set()
            for child in plan.children:
                result.update(
                    self._collect_direct_gp_indices(child)
                )
            return result

        raise KelschinatorCompileError(
            f"Unknown compiled plan type {type(plan).__name__}."
        )

    def _collect_gp_dependencies(
        self,
        meta: ProgramMeta,
        gidx: int,
        result: set[int],
    ) -> None:
        if gidx in result:
            return

        if gidx < 0 or gidx >= len(meta):
            raise KelschinatorCompileError(
                f"GP gene {gidx} is outside GP_meta."
            )

        result.add(gidx)

        for source_gidx in source_indices(
            meta.source[gidx]
        ):
            self._collect_gp_dependencies(
                meta,
                source_gidx,
                result,
            )

    def _compile_gp_instruction(
        self,
        meta: ProgramMeta,
        X: ProgramX,
        gidx: int,
    ) -> _GPInstruction:
        op = meta.op[gidx]
        sources = source_indices(meta.source[gidx])
        params = {
            str(key): _copy_param(value)
            for key, value in meta.params[gidx].items()
        }

        if op == "raw_input":
            return _GPInstruction(
                gidx=gidx,
                op=op,
                sources=(),
                params=params,
                output_slot=0,
                aux={},
            )

        signature = _meta_signature(meta, gidx)
        siblings = [
            index
            for index in range(len(meta))
            if _meta_signature(meta, index) == signature
        ]
        siblings.sort()

        try:
            output_slot = siblings.index(gidx)
        except ValueError as exc:
            raise KelschinatorCompileError(
                f"Cannot determine output slot for GP gene {gidx}."
            ) from exc

        aux: dict[str, Any] = {}

        # partition_composite has sample-set-dependent dynamic outputs.
        # Freeze the semantic color represented by this output slot.
        if op == "partition_composite":
            if output_slot % 2 == 0:
                color_gidx = siblings[output_slot]
                kind = "color"
            else:
                color_gidx = siblings[output_slot - 1]
                kind = "mask"

            color_values = [
                np.asarray(value).item()
                for value in X[color_gidx]
            ]

            if not color_values or any(
                value != color_values[0]
                for value in color_values[1:]
            ):
                raise KelschinatorCompileError(
                    "partition_composite color ID is not stable across training."
                )

            aux = {
                "kind": kind,
                "color": int(color_values[0]),
            }

        return _GPInstruction(
            gidx=gidx,
            op=op,
            sources=tuple(int(i) for i in sources),
            params=params,
            output_slot=output_slot,
            aux=aux,
        )

    def _eval_gp_gene(
        self,
        gidx: int,
        X_input: np.ndarray,
        instructions: dict[int, _GPInstruction],
        memo: dict[int, Any],
    ) -> Any:
        if gidx in memo:
            value = memo[gidx]
            return value.copy() if isinstance(value, np.ndarray) else value

        try:
            instruction = instructions[gidx]
        except KeyError as exc:
            raise RuntimeError(
                f"Distilled GP instruction {gidx} is missing."
            ) from exc

        if instruction.op == "raw_input":
            value = np.asarray(X_input).copy()
            memo[gidx] = value
            return value.copy()

        source_values = [
            self._eval_gp_gene(
                source,
                X_input,
                instructions,
                memo,
            )
            for source in instruction.sources
        ]

        if instruction.op == "partition_composite":
            if len(source_values) != 1:
                raise RuntimeError(
                    "partition_composite requires one source."
                )

            color = np.int64(instruction.aux["color"])

            if instruction.aux["kind"] == "color":
                value = color
            else:
                value = (
                    np.asarray(source_values[0]) == color
                ).astype(bool, copy=False)

        else:
            value = self._eval_registered_op(
                instruction,
                source_values,
            )

        if isinstance(value, np.ndarray):
            value = value.copy()

        memo[gidx] = value
        return value.copy() if isinstance(value, np.ndarray) else value

    def _eval_registered_op(
        self,
        instruction: _GPInstruction,
        source_values: list[Any],
    ) -> Any:
        try:
            info = OP_REGISTRY[instruction.op]
        except KeyError as exc:
            raise RuntimeError(
                f"Forward operation {instruction.op!r} is unavailable."
            ) from exc

        if info.func is None:
            raise RuntimeError(
                f"Forward operation {instruction.op!r} has no callable."
            )

        temp_meta = ProgramMeta(side="GP")
        temp_X = ProgramX(
            side="GP",
            sample_count=1,
        )

        temp_sources = []

        for index, source_value in enumerate(source_values):
            temp_gidx = temp_X.append_gene(
                [source_value]
            )
            temp_meta.append(
                source=-1,
                op=f"kelschinator_source_{index}",
                dims=_value_dims(source_value),
            )
            temp_sources.append(temp_gidx)

        if len(temp_sources) == 1:
            source_arg: Any = temp_sources[0]
        else:
            source_arg = tuple(temp_sources)

        result = info.func(
            temp_meta,
            temp_X,
            source_arg,
            **{
                key: _copy_param(value)
                for key, value in instruction.params.items()
            },
        )

        if isinstance(result, tuple):
            output_indices = tuple(int(i) for i in result)
        elif isinstance(result, list):
            output_indices = tuple(int(i) for i in result)
        else:
            output_indices = (int(result),)

        if instruction.output_slot >= len(output_indices):
            raise RuntimeError(
                f"Operation {instruction.op!r} produced "
                f"{len(output_indices)} outputs; distilled slot "
                f"{instruction.output_slot} is unavailable."
            )

        return temp_X[
            output_indices[instruction.output_slot],
            0,
        ]

    def _eval_plan(
        self,
        plan: Any,
        X_input: np.ndarray,
        instructions: dict[int, _GPInstruction],
        memo: dict[int, Any],
        context: dict[str, Any],
    ) -> Any:
        if isinstance(plan, _DirectPlan):
            source = self._eval_gp_gene(
                plan.gp_gidx,
                X_input,
                instructions,
                memo,
            )
            return self._apply_solution_rule(
                source,
                plan,
                context,
            )

        if isinstance(plan, _BundlePlan):
            return _BundleValue(
                tuple(
                    self._eval_plan(
                        child,
                        X_input,
                        instructions,
                        memo,
                        context,
                    )
                    for child in plan.children
                )
            )

        if isinstance(plan, _InversePlan):
            args: list[Any] = []

            for child_index, child in enumerate(plan.children):
                value = self._eval_plan(
                    child,
                    X_input,
                    instructions,
                    memo,
                    context,
                )

                if (
                    plan.inverse_op == "inv_partition_shape"
                    and child_index == 0
                    and isinstance(value, _BundleValue)
                    and len(value.values) == 2
                ):
                    context["root_output_shape"] = (
                        int(np.asarray(value.values[0]).item()),
                        int(np.asarray(value.values[1]).item()),
                    )

                if isinstance(value, _BundleValue):
                    args.extend(value.values)
                else:
                    args.append(value)

            inverse_info = INV_OP_REGISTRY[
                plan.inverse_op
            ]
            return inverse_info.func(*args)

        raise RuntimeError(
            f"Unknown plan type {type(plan).__name__}."
        )

    def _apply_solution_rule(
        self,
        source: Any,
        plan: _DirectPlan,
        context: dict[str, Any],
    ) -> Any:
        rule = plan.rule
        params = plan.params
        dtype = plan.target_dtype

        if plan.dims == 0:
            x = np.asarray(source).item()

            if rule == "identity":
                value = x
            elif rule == "negate":
                value = -x
            elif rule == "abs":
                value = abs(x)
            elif rule == "square":
                value = x * x
            elif rule == "add_constant":
                value = x + params["c"]
            elif rule == "constant_minus_x":
                value = params["c"] - x
            elif rule == "scale":
                value = params["c"] * x
            elif rule == "affine":
                value = (
                    params["a"] * x
                    + params["b"]
                )
            else:
                raise RuntimeError(
                    f"Unsupported scalar solution rule {rule!r}."
                )

            return np.asarray(value).astype(
                dtype,
                casting="unsafe",
            )[()]

        if plan.dims != 2:
            raise RuntimeError(
                f"Kelschinator direct rules currently support dims 0/2, "
                f"got {plan.dims}."
            )

        x = np.asarray(source)

        if rule == "identity":
            result = x

        elif rule in {
            "embed_zero_structural",
            "embed_zero_fixed",
        }:
            target_shape = self._target_shape_for_rule(
                x.shape,
                params,
                context,
            )

            if target_shape[0] < 0 or target_shape[1] < 0:
                raise RuntimeError(
                    "Distilled embedding produced invalid target shape."
                )

            if rule == "embed_zero_structural":
                row, col = _structural_offset(
                    params["position"],
                    target_shape,
                    x.shape,
                )
            else:
                row = int(params["row"])
                col = int(params["col"])

            result = np.zeros(
                target_shape,
                dtype=dtype,
            )
            h, w = x.shape

            if (
                row < 0
                or col < 0
                or row + h > target_shape[0]
                or col + w > target_shape[1]
            ):
                raise RuntimeError(
                    "Distilled embedding does not fit test target shape."
                )

            result[
                row:row + h,
                col:col + w,
            ] = x.astype(dtype, copy=False)

        elif rule in {
            "crop_structural",
            "crop_fixed",
        }:
            target_shape = self._target_shape_for_rule(
                x.shape,
                params,
                context,
            )

            if target_shape[0] < 0 or target_shape[1] < 0:
                raise RuntimeError(
                    "Distilled crop produced invalid target shape."
                )

            if rule == "crop_structural":
                row, col = _structural_offset(
                    params["position"],
                    x.shape,
                    target_shape,
                )
            else:
                row = int(params["row"])
                col = int(params["col"])

            th, tw = target_shape
            result = x[
                row:row + th,
                col:col + tw,
            ]

            if result.shape != target_shape:
                raise RuntimeError(
                    "Distilled crop does not fit test source."
                )

        elif rule == "shift_zero":
            result = _shift_zero(
                x,
                int(params["dr"]),
                int(params["dc"]),
                dtype,
            )

        elif rule == "tile":
            result = np.tile(
                x,
                (
                    int(params["rows"]),
                    int(params["cols"]),
                ),
            )

        else:
            raise RuntimeError(
                f"Unsupported matrix solution rule {rule!r}."
            )

        return np.asarray(result, dtype=dtype).copy()

    def _distill_2d_geometry(
        self,
        rule: str,
        params: dict[str, Any],
        gp_gene: np.ndarray,
        sp_gene: np.ndarray,
        *,
        root_gene: np.ndarray,
        is_root: bool,
    ) -> dict[str, Any]:
        params = dict(params)

        if rule not in {
            "embed_zero_structural",
            "embed_zero_fixed",
            "crop_structural",
            "crop_fixed",
        }:
            return params

        deltas = set()
        matches_root_shape = not is_root

        for sample_idx, (x_value, y_value) in enumerate(
            zip(gp_gene, sp_gene)
        ):
            x = np.asarray(x_value)
            y = np.asarray(y_value)

            if x.ndim != 2 or y.ndim != 2:
                raise KelschinatorCompileError(
                    f"2D solution rule {rule!r} received non-2D training data."
                )

            root_value = np.asarray(root_gene[sample_idx])
            if y.shape != root_value.shape:
                matches_root_shape = False

            deltas.add(
                (
                    y.shape[0] - x.shape[0],
                    y.shape[1] - x.shape[1],
                )
            )

        if matches_root_shape:
            params["_target_shape_mode"] = "root"
            return params

        if len(deltas) != 1:
            raise KelschinatorCompileError(
                f"2D rule {rule!r} has neither root-shape semantics "
                "nor one executable shape delta."
            )

        params["_target_shape_mode"] = "delta"
        params["shape_delta"] = next(iter(deltas))
        return params

    def _target_shape_for_rule(
        self,
        source_shape: tuple[int, int],
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[int, int]:
        mode = params.get("_target_shape_mode", "delta")

        if mode == "root":
            shape = context.get("root_output_shape")
            if shape is None:
                raise RuntimeError(
                    "A 2D solution requires the solved root output shape "
                    "before it can be reconstructed."
                )
            return (int(shape[0]), int(shape[1]))

        if mode != "delta":
            raise RuntimeError(
                f"Unknown target-shape mode {mode!r}."
            )

        dh, dw = params["shape_delta"]
        return (
            source_shape[0] + int(dh),
            source_shape[1] + int(dw),
        )

    def _gene_dtype(
        self,
        gene: np.ndarray,
        *,
        expected_dims: int | None,
    ) -> np.dtype:
        dtypes: set[np.dtype] = set()

        for value in gene:
            array = np.asarray(value)

            if (
                expected_dims is not None
                and array.ndim != expected_dims
            ):
                raise KelschinatorCompileError(
                    "SP target gene dimensionality is inconsistent."
                )

            dtypes.add(np.dtype(array.dtype))

        if len(dtypes) != 1:
            raise KelschinatorCompileError(
                "SP target gene does not have one stable dtype."
            )

        return next(iter(dtypes))

    def _describe_pipeline(
        self,
        root_plan: Any,
        instructions: dict[int, _GPInstruction],
    ) -> tuple[str, ...]:
        steps = []

        for gidx in sorted(instructions):
            instruction = instructions[gidx]

            if instruction.op == "raw_input":
                steps.append(f"GP[{gidx}] <- input")
            else:
                steps.append(
                    f"GP[{gidx}] <- {instruction.op}"
                    f"{instruction.sources}"
                )

        def describe_plan(plan: Any, prefix: str = "ST") -> None:
            if isinstance(plan, _DirectPlan):
                steps.append(
                    f"{prefix} <- GP[{plan.gp_gidx}] "
                    f"via {plan.rule}"
                )
                return

            if isinstance(plan, _BundlePlan):
                for index, child in enumerate(plan.children):
                    describe_plan(
                        child,
                        f"{prefix}.and[{index}]",
                    )
                return

            if isinstance(plan, _InversePlan):
                for index, child in enumerate(plan.children):
                    describe_plan(
                        child,
                        f"{prefix}.{plan.inverse_op}[{index}]",
                    )
                steps.append(
                    f"{prefix} <- {plan.inverse_op}(...)"
                )
                return

        describe_plan(root_plan)
        return tuple(steps)
