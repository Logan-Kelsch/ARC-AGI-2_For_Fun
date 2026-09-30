from dataclasses import dataclass, field
import ops.grid_ops as grid_ops
import numpy as np

def _copy_eval_value(value):
    """Copy heterogeneous evaluation values without coercing their structure."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            copied = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                copied[index] = _copy_eval_value(value[index])
            return copied
        return value.copy()

    if isinstance(value, list):
        return [_copy_eval_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_copy_eval_value(item) for item in value)

    return value


def _pack_eval_targets(values) -> np.ndarray:
    """Store heterogeneous per-sample targets without NumPy coercion."""
    packed = np.empty(len(values), dtype=object)

    for sample_idx, value in enumerate(values):
        packed[sample_idx] = _copy_eval_value(value)

    return packed


@dataclass
class GP_EvalNode:
    """One retained target location in the GP_Set evaluation tree.

    target always preserves the full original output-side representation.

    remaining_target is optional mutable pool state. It is used when a node can
    be partitioned into independently discovered subsets, such as color_id and
    color_presence. Extracting a subset changes remaining_target but never
    destroys target, so parent/full-output targets remain testable.

    A discovered answer is terminal when that exact target/subset should no
    longer be offered to later evaluation passes.
    """

    name: str
    target: np.ndarray
    answer_present: bool = False
    gene_idx: int = -1
    solution_number: int = -1
    terminal: bool = False
    children: dict[str, "GP_EvalNode"] = field(default_factory=dict)
    remaining_target: np.ndarray | None = None
    is_pool: bool = False
    dynamic_terminal: bool = False
    source_nodes: tuple[str, ...] = ()

    def add_child(self, child: "GP_EvalNode") -> "GP_EvalNode":
        if child.name in self.children:
            raise ValueError(
                f"Evaluation node {self.name!r} already has child "
                f"{child.name!r}."
            )
        self.children[child.name] = child
        return child

    def __getitem__(self, name: str) -> "GP_EvalNode":
        return self.children[name]

    def find(self, name: str) -> "GP_EvalNode":
        if self.name == name:
            return self

        for child in self.children.values():
            try:
                return child.find(name)
            except KeyError:
                pass

        raise KeyError(f"No GP evaluation node named {name!r}.")

    def walk(self):
        yield self
        for child in self.children.values():
            yield from child.walk()

    @property
    def sample_count(self) -> int:
        return len(self.target)

    @property
    def searchable(self) -> bool:
        return not self.terminal

    def sample_target(self, sample_idx: int, *, remaining: bool = False):
        values = (
            self.remaining_target
            if remaining and self.remaining_target is not None
            else self.target
        )
        return values[sample_idx]

    def _mark_terminal(
        self,
        *,
        gene_idx: int,
        solution_number: int,
    ) -> None:
        gene_idx = int(gene_idx)
        solution_number = int(solution_number)

        if gene_idx < 0:
            raise ValueError("gene_idx must be non-negative.")
        if solution_number < 1:
            raise ValueError("solution_number must be >= 1.")
        if self.terminal:
            raise ValueError(
                f"Evaluation node {self.name!r} is already terminal."
            )

        self.answer_present = True
        self.gene_idx = gene_idx
        self.solution_number = solution_number
        self.terminal = True

    def clear_answer(self) -> None:
        self.answer_present = False
        self.gene_idx = -1
        self.solution_number = -1
        self.terminal = False

    def reset_remaining_target(self) -> None:
        if not self.is_pool:
            self.remaining_target = None
            return

        self.remaining_target = _pack_eval_targets(
            [_copy_eval_value(value) for value in self.target]
        )


@dataclass
class GP_EvalTree:
    """Persistent output-target decomposition owned by a GP_Set.

    Structure:

        root
        ├── shape
        │   ├── h
        │   └── w
        └── composite
            ├── color_id          (pool)
            ├── color_presence    (pool)
            └── dynamically extracted terminal subsets

    Solution numbering is global to this tree and starts at 1. The tree only
    records discoveries; it does not search GP_Set.data for them.
    """

    root: GP_EvalNode
    next_solution_number: int = 1

    def __getitem__(self, name: str) -> GP_EvalNode:
        return self.root.find(name)

    @property
    def shape(self) -> GP_EvalNode:
        return self["shape"]

    @property
    def h(self) -> GP_EvalNode:
        return self["h"]

    @property
    def w(self) -> GP_EvalNode:
        return self["w"]

    @property
    def composite(self) -> GP_EvalNode:
        return self["composite"]

    @property
    def color_id(self) -> GP_EvalNode:
        return self["color_id"]

    @property
    def color_presence(self) -> GP_EvalNode:
        return self["color_presence"]

    @property
    def sample_count(self) -> int:
        return self.root.sample_count

    def walk(self):
        return self.root.walk()

    @property
    def terminal_nodes(self) -> tuple[GP_EvalNode, ...]:
        """Discovered terminals in chronological solution order."""
        return tuple(
            sorted(
                (
                    node
                    for node in self.walk()
                    if node.terminal and node.solution_number >= 1
                ),
                key=lambda node: node.solution_number,
            )
        )

    def searchable_nodes(self) -> tuple[GP_EvalNode, ...]:
        """Nodes available to a future evaluator.

        A terminal node closes its own branch, so its descendants are not
        returned. Ancestors remain searchable and can still be discovered later.
        Dynamic terminal subset nodes are retained for history but never
        re-enter the search.
        """
        nodes: list[GP_EvalNode] = []

        def visit(node: GP_EvalNode) -> None:
            if node.terminal:
                return

            nodes.append(node)

            for child in node.children.values():
                visit(child)

        visit(self.root)
        return tuple(nodes)

    def _resolve_node(self, node_or_name: GP_EvalNode | str) -> GP_EvalNode:
        if isinstance(node_or_name, GP_EvalNode):
            return node_or_name
        if isinstance(node_or_name, str):
            return self[node_or_name]
        raise TypeError("Expected a GP_EvalNode or node name.")

    def mark_terminal_solution(
        self,
        node_or_name: GP_EvalNode | str,
        gene_idx: int,
    ) -> GP_EvalNode:
        """Chronologically record an exact discovered node solution."""
        node = self._resolve_node(node_or_name)

        node._mark_terminal(
            gene_idx=gene_idx,
            solution_number=self.next_solution_number,
        )
        self.next_solution_number += 1

        return node

    def add_terminal_subset(
        self,
        parent_or_name: GP_EvalNode | str,
        *,
        name: str,
        target,
        gene_idx: int,
        source_nodes: tuple[str, ...] = (),
    ) -> GP_EvalNode:
        """Retain any newly discovered subset as a terminal child.

        This is the generic foundation for future partitions beyond colors.
        The parent remains non-terminal and therefore remains testable as a
        complete representation.
        """
        parent = self._resolve_node(parent_or_name)
        gene_idx = int(gene_idx)

        if gene_idx < 0:
            raise ValueError("gene_idx must be non-negative.")
        if parent.terminal:
            raise ValueError(
                f"Cannot add a subset beneath terminal node {parent.name!r}."
            )
        if name in parent.children:
            raise ValueError(
                f"Evaluation node {parent.name!r} already has child {name!r}."
            )

        packed_target = (
            target
            if isinstance(target, np.ndarray)
            and target.dtype == object
            and target.ndim == 1
            else _pack_eval_targets(list(target))
        )

        if len(packed_target) != self.sample_count:
            raise ValueError(
                "Subset target must contain one value per training sample."
            )

        terminal = parent.add_child(
            GP_EvalNode(
                name=name,
                target=_pack_eval_targets(
                    [_copy_eval_value(value) for value in packed_target]
                ),
                dynamic_terminal=True,
                source_nodes=tuple(source_nodes),
            )
        )

        return self.mark_terminal_solution(terminal, gene_idx)

    def extract_color_terminal(
        self,
        color: int,
        gene_idx: int,
        *,
        name: str | None = None,
    ) -> GP_EvalNode:
        """Extract one color ID + presence subset into a terminal solution.

        The original color_id/color_presence targets and the full composite
        target remain unchanged. Only each pool's remaining_target is reduced.

        Per sample, the extracted target is:
            [array([color]), one-channel presence]
        when the color occurs, otherwise:
            [empty color array, empty presence channel]

        This preserves an exact across-sample representation, including absence.
        """
        color = int(color)
        node_name = name or f"color_{color}"

        if self.color_id.terminal or self.color_presence.terminal:
            raise ValueError(
                "Cannot extract a color subset from a terminal color pool."
            )
        if node_name in self.composite.children:
            raise ValueError(
                f"Composite already contains child {node_name!r}."
            )

        color_ids_remaining = self.color_id.remaining_target
        presence_remaining = self.color_presence.remaining_target

        if color_ids_remaining is None or presence_remaining is None:
            raise ValueError("Color evaluation pools are not initialized.")

        next_ids = _pack_eval_targets(
            [_copy_eval_value(value) for value in color_ids_remaining]
        )
        next_presence = _pack_eval_targets(
            [_copy_eval_value(value) for value in presence_remaining]
        )

        extracted_targets = []
        found_any = False

        for sample_idx in range(self.sample_count):
            ids = np.asarray(next_ids[sample_idx], dtype=int)
            presence = np.asarray(next_presence[sample_idx], dtype=bool)

            if presence.ndim != 3:
                raise ValueError(
                    "color_presence remaining targets must be 3D arrays."
                )
            if len(ids) != presence.shape[0]:
                raise ValueError(
                    "color_id and color_presence pools are misaligned."
                )

            matches = np.flatnonzero(ids == color)

            if len(matches) > 1:
                raise ValueError(
                    f"Color {color} appears more than once in sample "
                    f"{sample_idx}'s color pool."
                )

            subset = np.empty(2, dtype=object)

            if len(matches) == 1:
                found_any = True
                index = int(matches[0])

                subset[0] = np.asarray([color], dtype=int)
                subset[1] = presence[index:index + 1].copy()

                next_ids[sample_idx] = np.delete(ids, index)
                next_presence[sample_idx] = np.delete(
                    presence,
                    index,
                    axis=0,
                )
            else:
                subset[0] = np.empty(0, dtype=int)
                subset[1] = np.empty(
                    (0, presence.shape[1], presence.shape[2]),
                    dtype=bool,
                )

            extracted_targets.append(subset)

        if not found_any:
            raise ValueError(
                f"Color {color} is not available in the remaining color pools."
            )

        terminal = self.add_terminal_subset(
            self.composite,
            name=node_name,
            target=_pack_eval_targets(extracted_targets),
            gene_idx=gene_idx,
            source_nodes=("color_id", "color_presence"),
        )

        self.color_id.remaining_target = next_ids
        self.color_presence.remaining_target = next_presence

        return terminal

    def clear_answers(self) -> None:
        """Reset discovery history and restore all mutable pools."""
        def reset(node: GP_EvalNode) -> None:
            dynamic_names = [
                name
                for name, child in node.children.items()
                if child.dynamic_terminal
            ]
            for name in dynamic_names:
                del node.children[name]

            node.clear_answer()
            node.reset_remaining_target()

            for child in node.children.values():
                reset(child)

        reset(self.root)
        self.next_solution_number = 1


def init_gp_eval_tree(outputs) -> GP_EvalTree:
    """Build the exact output-side target decomposition for a GP_Set.

    No attempt is made to locate answers in GP_Set.data. This only establishes
    the target structure and mutable subset pools used by later evaluation.
    """
    outputs = list(outputs)
    if not outputs:
        raise ValueError("Cannot build a GP evaluation tree with no outputs.")

    root_targets = []
    shape_targets = []
    h_targets = []
    w_targets = []
    composite_targets = []
    color_id_targets = []
    color_presence_targets = []

    for sample_idx, output in enumerate(outputs):
        output = np.asarray(output)
        if output.ndim != 2:
            raise ValueError(
                f"outputs[{sample_idx}] must be a 2D grid, "
                f"got shape {output.shape}."
            )

        shape, colors, presence = grid_ops.grid_dissection(
            output,
            as_seperate=True,
        )

        composite = np.empty(2, dtype=object)
        composite[0] = np.asarray(colors).copy()
        composite[1] = np.asarray(presence, dtype=bool).copy()

        root_targets.append(output.copy())
        shape_targets.append(np.asarray(shape, dtype=int).copy())
        h_targets.append(int(shape[0]))
        w_targets.append(int(shape[1]))
        composite_targets.append(composite)
        color_id_targets.append(np.asarray(colors).copy())
        color_presence_targets.append(
            np.asarray(presence, dtype=bool).copy()
        )

    root = GP_EvalNode(
        name="root",
        target=_pack_eval_targets(root_targets),
    )

    shape_node = root.add_child(
        GP_EvalNode(
            name="shape",
            target=_pack_eval_targets(shape_targets),
        )
    )
    shape_node.add_child(
        GP_EvalNode(
            name="h",
            target=_pack_eval_targets(h_targets),
        )
    )
    shape_node.add_child(
        GP_EvalNode(
            name="w",
            target=_pack_eval_targets(w_targets),
        )
    )

    composite_node = root.add_child(
        GP_EvalNode(
            name="composite",
            target=_pack_eval_targets(composite_targets),
        )
    )

    color_id_target = _pack_eval_targets(color_id_targets)
    color_presence_target = _pack_eval_targets(color_presence_targets)

    composite_node.add_child(
        GP_EvalNode(
            name="color_id",
            target=color_id_target,
            remaining_target=_pack_eval_targets(color_id_targets),
            is_pool=True,
        )
    )
    composite_node.add_child(
        GP_EvalNode(
            name="color_presence",
            target=color_presence_target,
            remaining_target=_pack_eval_targets(color_presence_targets),
            is_pool=True,
        )
    )

    return GP_EvalTree(root=root)


@dataclass
class GP_Set:
    input: np.ndarray
    output: np.ndarray 
    data: np.ndarray      # dtype=object
    op: np.ndarray        # strings
    source: np.ndarray    # int or collection of source gene indices
    status: np.ndarray    # strings
    eval_tree: GP_EvalTree | None = None

    @classmethod
    def empty(cls, L: int):
        return cls(
            input=np.empty(L, dtype=list),
            output=np.empty(L, dtype=list),
            data=np.full(L, np.empty(L, dtype=object), dtype=np.ndarray),
            op=[],
            source=[],
            status=[],#null, unif, uniq, part
        )

    def __len__(self):
        return len(self.data)

    def get_essential_gidx(self):
        """Return gene indices directly used by retained evaluation solutions."""
        return get_essential_gidx(self)

    def get_essential_gidx_tree(self):
        """Return solution genes plus every recursive source dependency."""
        return get_essential_gidx_tree(self)


def get_essential_gidx(gp_set):
    """Return unique gene indices directly used by evaluation-tree solutions.

    Solution order follows GP_EvalTree.solution_number. If multiple retained
    solutions use the same gene index, that index is returned only once at its
    first occurrence.
    """
    if gp_set.eval_tree is None:
        raise ValueError("GP_Set has no evaluation tree.")

    essential = []
    seen = set()

    solution_nodes = sorted(
        (
            node
            for node in gp_set.eval_tree.walk()
            if node.answer_present and node.gene_idx >= 0
        ),
        key=lambda node: (
            node.solution_number
            if node.solution_number >= 1
            else float("inf")
        ),
    )

    for node in solution_nodes:
        gene_idx = int(node.gene_idx)

        if gene_idx not in seen:
            essential.append(gene_idx)
            seen.add(gene_idx)

    return essential


def _source_gidx_list(source_value):
    """Normalize one GP_Set.source entry into a flat list of source indices.

    A source may be:
      - -1 for a raw/source-free gene,
      - one integer gene index,
      - a list/tuple/ndarray of multiple source indices,
      - nested combinations of those containers.

    -1 entries terminate that dependency branch and are not returned.
    """
    if isinstance(source_value, np.generic):
        source_value = source_value.item()

    if isinstance(source_value, np.ndarray):
        if source_value.ndim == 0:
            return _source_gidx_list(source_value.item())
        source_value = source_value.tolist()

    if isinstance(source_value, (list, tuple)):
        result = []
        seen = set()

        for item in source_value:
            for gene_idx in _source_gidx_list(item):
                if gene_idx not in seen:
                    result.append(gene_idx)
                    seen.add(gene_idx)

        return result

    if isinstance(source_value, bool):
        raise TypeError("GP source indices may not be booleans.")

    if isinstance(source_value, (int, np.integer)):
        source_idx = int(source_value)

        if source_idx == -1:
            return []
        if source_idx < -1:
            raise ValueError(
                f"GP source index must be -1 or non-negative, got {source_idx}."
            )

        return [source_idx]

    raise TypeError(
        "GP source entries must be an integer, -1, or a nested "
        "list/tuple/ndarray of integer indices."
    )


def get_essential_gidx_tree(gp_set):
    """Return the dependency-complete minimal gene-index reconstruction set.

    Starts from get_essential_gidx(gp_set), then recursively follows
    gp_set.source[gene_idx] until every branch terminates at a source entry of
    -1.

    Multiple-source genes are fully expanded. Returned indices are unique and
    dependency-first: every source gene appears before any gene that depends on
    it. This makes the output suitable for reconstructing a minimal GP program
    in executable order.
    """
    roots = get_essential_gidx(gp_set)

    if not roots:
        return []

    source_count = len(gp_set.source)
    ordered = []
    visited = set()
    visiting = set()

    def visit(gene_idx):
        gene_idx = int(gene_idx)

        if gene_idx in visited:
            return

        if gene_idx in visiting:
            raise ValueError(
                f"Cycle detected in GP source dependencies at gene {gene_idx}."
            )

        if gene_idx < 0 or gene_idx >= source_count:
            raise IndexError(
                f"Gene index {gene_idx} has no matching GP_Set.source entry "
                f"(source length={source_count})."
            )

        visiting.add(gene_idx)

        for source_idx in _source_gidx_list(gp_set.source[gene_idx]):
            visit(source_idx)

        visiting.remove(gene_idx)
        visited.add(gene_idx)
        ordered.append(gene_idx)

    for root_idx in roots:
        visit(root_idx)

    return ordered


def init_gp_mat(grid_set):
    #we will be recieving the training grid
    #we will resolve associations along inputs
    #we will maybe resolve associations between outputs
    #and we will resolve associations to explain 100% of information
    #   of an output from an input, according to the provided data
    #we can rate risk of error by quantifying specificity/particularity 
    #   of infromation used within associations
    L = len(grid_set)
    gp_mat = GP_Set.empty(L)
    for i in range(L):
        gp_mat.input[i] = grid_set[i].input
        gp_mat.output[i] = grid_set[i].output
        gp_mat.data[i] = np.empty(0, dtype=object)


    for i in range(len(grid_set)):
        item = np.empty(4, dtype=object)
        item[0] = np.asarray(grid_set[i].input)
        item[1], item[2], item[3] = grid_ops.grid_dissection(grid_set[i].input, as_seperate=True)
        gp_mat.data[i] = np.concatenate([gp_mat.data[i], item]) 
    #NOTE temp solutions for initial stats
    #identity composite
    gp_mat.op.append('NULL')
    gp_mat.source.append(-1)
    gp_mat.status.append('NULL')
    #identity shape
    gp_mat.op.append('shape extract')
    gp_mat.source.append(0)
    gp_mat.status.append('NULL')
    #identity colors used (color ID)
    gp_mat.op.append('color ID - partition')
    gp_mat.source.append(0)
    gp_mat.status.append('NULL')
    #identity colors used (boolean presence)
    gp_mat.op.append('color presence - partition')
    gp_mat.source.append(0)
    gp_mat.status.append('NULL')
    

    # Retain the exact output decomposition that future GP evaluation will
    # attempt to explain with gene columns across every training sample.
    gp_mat.eval_tree = init_gp_eval_tree(gp_mat.output)

    return gp_mat


def _gp_values_equal(a, b):
    """Exact structural/content equality for heterogeneous GP gene values."""
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()

    if isinstance(a, np.ndarray):
        if a.ndim == 0:
            a = a.item()
        else:
            a = a.tolist()

    if isinstance(b, np.ndarray):
        if b.ndim == 0:
            b = b.item()
        else:
            b = b.tolist()

    a_is_seq = isinstance(a, (list, tuple))
    b_is_seq = isinstance(b, (list, tuple))

    if a_is_seq or b_is_seq:
        if not (a_is_seq and b_is_seq):
            return False
        if len(a) != len(b):
            return False
        return all(_gp_values_equal(x, y) for x, y in zip(a, b))

    try:
        result = a == b
    except Exception:
        return False

    if isinstance(result, np.ndarray):
        return bool(np.all(result))

    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _gp_status_is_null(status):
    """Return True only for unresolved GP statuses."""
    if status is None:
        return True
    return isinstance(status, str) and status.strip().lower() == "null"


def gp_fill_status(gp_set):
    """Fill unresolved gene statuses by exact equivalence across all samples.

    For every gene index whose current status is None or "NULL", compare:

        gp_set.data[0][gene_index]
        gp_set.data[1][gene_index]
        ...
        gp_set.data[L - 1][gene_index]

    No distance, similarity, or partial-match measure is used. Values either
    belong to the same exact equivalence class or they do not.

    Status assignment:
        unif : all sample values are equivalent (1 equivalence class)
        uniq : every sample value is different (L equivalence classes)
        part : some values repeat and some differ (2..L-1 classes)

    Lists/tuples and NumPy ndarrays are compared recursively by structure and
    content, so equivalent list/ndarray representations compare equal.
    Nested object arrays are supported.

    The GP_Set is mutated in place and returned.
    """
    sample_count = len(gp_set.data)
    if sample_count == 0:
        raise ValueError("gp_set.data contains no samples.")

    unresolved = [
        gene_index
        for gene_index, status in enumerate(gp_set.status)
        if _gp_status_is_null(status)
    ]

    for gene_index in unresolved:
        values = []

        for sample_index in range(sample_count):
            sample_genes = gp_set.data[sample_index]

            try:
                value = sample_genes[gene_index]
            except (IndexError, TypeError) as exc:
                raise ValueError(
                    f"Sample {sample_index} does not contain gene index "
                    f"{gene_index}."
                ) from exc

            values.append(value)

        representatives = []
        for value in values:
            if not any(
                _gp_values_equal(value, representative)
                for representative in representatives
            ):
                representatives.append(value)

        class_count = len(representatives)

        if class_count == 1:
            gp_set.status[gene_index] = "unif"
        elif class_count == sample_count:
            gp_set.status[gene_index] = "uniq"
        else:
            gp_set.status[gene_index] = "part"

    return gp_set

def disc_fit_v1(task_train):
    #CONCEPT ITER 3
    #this search process finds an abstract IA that has zero variance at full resolution
    #   therefore this model would be able to interpret attributes from any input
    #   and be able to model output
    #this means the following of our base case (step zero):
    #   - contains zero known uniformities
    #   - has variance equal to estimated maximum test variance
    #this means the following of taking a step:
    #   - taking a step consists of:
    #       - identifying new IA, or
    #       - testing for uniformity
    #       - partitioning an uncertain space?
    #   - we accept a step if step's uniformity constricts variance
    #scratch thought process within below window ----------
    #scratch all this, we need to be instead looking at level of constraint on certainty
    #this means the following of evaluating variance:
    #   - variance in shape has natural priority over presence and color
    #   - variance in presence has natural priority over color
    #   - therefore we have a defined hierarchy of loss as:
    #       - shape > presence ~=~ color
    #   - first iteration of loss can be rough:
    #        L(shape) if >0 else L(presence + color)
    #   L(y-hat shape) = ln(y-hat/y)^2
    #scratch thought process within above window ----------
    #scratch all this, we need to be instead looking at level of constraint on certainty
    #that means error will instead be on distance from zero constraint to full constraint
    #this must be a normalized problem since we have two bounds:
    #   - fully constrained and fully unknown
    # an example could be L(model) = L(shape) + L(structure)
    #   where L(shape) is 1 when we have no policy and 0 when we have it fully modeled
    #   where L(structure) is 0.5 when we have partitioned the problem and solved half
    #       ex: defined color 0 as background, and now exploring "foreground"



    #CONCEPT ITER 2
    #this idea came to me moments before falling asleep.
    #this is somewhat of an specific tree walk where we walk deep and back to find
    #   abstract PASS cases of uniformity before we can predict pixel and color.
    #in the simplest case, this walk may begin with seeing that:
    #   shape usage has unifromity across sample inputs,
    #   color usage has uniformity across sample inputs,
    #   pixel usage has uniformity across sample inputs,
    #   shape usage has uniformity across in/outs,
    #   color usage has uniformity across in/outs, or
    #   pixel usage has unifromity across in/outs.
    #uniformity in these would mean we can use minimal info to constrict model variance
    #we ultimately are walking this tree until we push the answer variance to zero.



    #CONCEPT ITER 1
    #first off we will need to understand we are dealing with a
    #   tree based walking evaluation problem
    #so we need a universal evaluation function
    #this function justs tests for uniformity along all train cases
    #   including or not including gang violence RIP CK
    #   including or not including the output cases
    #that means we also need a step function to step deeper into observation
    #that also means we need a memory variable that maintains what we know
    #that means we need to define a discrete checklist of what
    #   we need to know to answer the question

    pass

def test_uniformity():
    #we must be able to make a function that answers
    #   whether or not some attributes or IA are uniform across samples
    pass
    

