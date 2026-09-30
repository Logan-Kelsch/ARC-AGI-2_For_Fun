from dataclasses import dataclass, field
import ops.grid_ops as grid_ops
import numpy as np

@dataclass
class GP_EvalNode:
    """One target location in the GP_Set evaluation tree.

    target is an outer object array with one exact target value per training
    sample. A future GP evaluator will compare one GP gene column against this
    target vector across every sample.

    answer_present and gene_idx are intentionally only bookkeeping fields here.
    This module does not search for matching genes yet.
    """

    name: str
    target: np.ndarray
    answer_present: bool = False
    gene_idx: int = -1
    children: dict[str, "GP_EvalNode"] = field(default_factory=dict)

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

    def sample_target(self, sample_idx: int):
        return self.target[sample_idx]

    def mark_answer(self, gene_idx: int) -> None:
        """Associate an exact across-sample GP gene with this target node."""
        gene_idx = int(gene_idx)
        if gene_idx < 0:
            raise ValueError("gene_idx must be non-negative.")

        self.answer_present = True
        self.gene_idx = gene_idx

    def clear_answer(self) -> None:
        self.answer_present = False
        self.gene_idx = -1


@dataclass
class GP_EvalTree:
    """Retained output-target decomposition owned by a GP_Set.

    Structure:

        root
        ├── shape
        │   ├── h
        │   └── w
        └── composite
            ├── color_id
            └── color_presence

    Every node stores the exact target representation for every training sample.
    Future evaluation code can mark nodes when a GP gene column exactly matches
    that target vector across all samples.
    """

    root: GP_EvalNode

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

    def clear_answers(self) -> None:
        for node in self.walk():
            node.clear_answer()


def _pack_eval_targets(values) -> np.ndarray:
    """Store heterogeneous per-sample targets without NumPy coercion."""
    packed = np.empty(len(values), dtype=object)

    for sample_idx, value in enumerate(values):
        if isinstance(value, np.ndarray):
            packed[sample_idx] = value.copy()
        else:
            packed[sample_idx] = value

    return packed


def init_gp_eval_tree(outputs) -> GP_EvalTree:
    """Build the exact output-side target decomposition for a GP_Set.

    No attempt is made to locate answers in GP_Set.data. This only establishes
    the target structure that a later evaluation/search function will compare
    against the gene matrix.
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
    composite_node.add_child(
        GP_EvalNode(
            name="color_id",
            target=_pack_eval_targets(color_id_targets),
        )
    )
    composite_node.add_child(
        GP_EvalNode(
            name="color_presence",
            target=_pack_eval_targets(color_presence_targets),
        )
    )

    return GP_EvalTree(root=root)


@dataclass
class GP_Set:
    input: np.ndarray
    output: np.ndarray 
    data: np.ndarray      # dtype=object
    op: np.ndarray        # strings
    source: np.ndarray    # ints
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
    

