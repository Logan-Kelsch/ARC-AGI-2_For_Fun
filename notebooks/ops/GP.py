from dataclasses import dataclass
import numpy as np

@dataclass
class GP_Set:
    input: np.ndarray
    output: np.ndarray 
    data: np.ndarray      # dtype=object
    op: np.ndarray        # strings
    source: np.ndarray    # ints
    status: np.ndarray    # strings

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
        item = np.empty(1, dtype=object)
        item[0] = np.asarray(grid_set[i].input)
        gp_mat.data[i] = np.concatenate([gp_mat.data[i], item]) 
    gp_mat.op.append('NULL')
    gp_mat.source.append(-1)
    gp_mat.status.append('NULL')

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
    

def grid_dissection(grid):
    
    pass