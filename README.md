# ARC-AGI-2 For Fun

An experimental **Genetic Programming / program-synthesis approach to ARC-AGI-2**.

The central idea of this repository is not to directly train a model to predict an output grid. Instead, the solver incrementally builds a library of interpretable **genes** describing each training example, discovers exact relationships among those genes, and searches for small programs that progressively explain the output.

This is an active research project. The architecture below describes the current working hypothesis and will change as the search process is refined.

---

## Problem framing

An ARC task contains a small number of input/output demonstration pairs and one or more unseen test inputs.

The objective is to infer a transformation:

~~~text
input grid -> output grid
~~~

that exactly reproduces all demonstration outputs and then generalizes to the test input.

Rather than treating this as a monolithic prediction problem, this project decomposes the problem into two interacting spaces:

~~~text
INTERPRETATION SPACE                LOSS SPACE

raw input                           unsolved output
   |                                     |
   v                                     v
genes / attributes                  loss tree
   |                                     |
   v                                     v
derived genes                       unresolved subproblems
   |                                     |
   +--------------- search --------------+
                     |
                     v
             discovered program
~~~

The GP search expands the interpretation space while the loss tree tells the search **what still needs to be explained**.

---

## 1. Gene matrix

The foundational data structure is **GP_Set**, initialized with:

~~~python
gp_mat = init_gp_mat(task_train)
~~~

There is one row of gene data for each training example.

Conceptually:

~~~text
                 gene 0      gene 1      gene 2      ...      gene G

sample 0          x00         x01         x02                  x0G
sample 1          x10         x11         x12                  x1G
sample 2          x20         x21         x22                  x2G
  ...
sample L          xL0         xL1         xL2                  xLG
~~~

Each gene index represents the **same interpretation or operation across every training example**.

A gene may contain almost any structured value useful for reasoning about the task:

- integers or floats
- strings
- 1D or 2D lists
- NumPy arrays
- boolean masks
- output/input dimensions
- color sets
- object/component descriptions
- nested combinations of these structures

The GP structure also records metadata for each gene:

~~~text
data    actual interpreted value
op      operation that generated the gene
source  parent/source gene
status  relationship of this gene across training samples
~~~

The initial gene is the raw input grid. Future operations append new genes derived from existing ones.

---

## 2. Exact equivalence across demonstrations

Before reasoning about how different values relate, the system first asks a simple set-theoretic question:

> For the same gene index, which training examples contain exactly equivalent values?

**gp_fill_status(gp_set)** classifies unresolved genes as:

~~~text
unif   all training examples contain the same value
uniq   every training example contains a different value
part   some examples are identical while others differ
~~~

For example:

~~~text
[5, 5, 5] -> unif
[5, 7, 9] -> uniq
[5, 5, 9] -> part
~~~

This comparison is based only on **exact structural/content equality**.

There is no similarity score and no attempt to quantify how different two values are.

Nested lists, normal NumPy arrays, object arrays, matrices, and mixed nested structures can all participate in these equivalence tests.

This creates partitions of the training examples that can later become useful evidence for deterministic relationships.

---

## 3. Gene expansion

Genetic Programming operations transform existing genes into new genes.

Conceptually:

~~~text
gene_i
   |
   +-- operation A --> gene_j
   |
   +-- operation B --> gene_k
   |
   '-- operation C --> gene_m
~~~

Examples may eventually include operations such as:

- grid dissection
- dimensions
- colors present
- color removal/addition
- masks
- connected components
- counts
- bounding boxes
- spatial transformations
- relations between components
- transformations between input and output attributes

Each generated value is stored at the same gene index across all training examples.

The important distinction is that the system is not limited to searching for **constant genes**.

A useful rule may connect genes whose values vary across every demonstration.

For example:

~~~text
input colors  {0, 1, 5} -> output colors {0, 5}
input colors  {0, 1, 7} -> output colors {0, 7}
input colors  {0, 1, 3} -> output colors {0, 3}
~~~

The raw color sets are not uniform, but the relationship

~~~text
output_colors = input_colors - {1}
~~~

is uniform.

The long-term search therefore operates over both **gene values** and **relations between genes**.

---

## 4. Loss is a tree, not a scalar

The current evaluator deliberately avoids reducing output error to one number.

Instead:

~~~python
tree = loss_resolution(yhat, y)
~~~

creates:

~~~text
root
├── shape
│   ├── h
│   └── w
└── composite
~~~

The root represents the complete ARC output problem. Its children are a full decomposition of that problem.

### Shape

Output dimensions are handled separately:

~~~text
shape
├── h
└── w
~~~

Height and width are terminal loss nodes.

This makes output dimensions high-level constraints: before the exact contents of an output can be fully specified, the solver must determine the space in which those contents exist.

### Composite grid loss

The **composite** node represents the categorical state of every output pixel.

Its value is an **11 x 11 color-transition matrix**.

States:

~~~text
0..9  ARC colors
10    INVALID
~~~

Rows represent the predicted state and columns represent the target state.

For example:

~~~python
matrix[3, 7]
~~~

is the number of positions predicted as color 3 that should have been color 7.

A correct prediction contains transitions only on the normal color diagonal.

The **INVALID** state handles shape mismatches:

~~~text
INVALID -> target color
    target contains a pixel that does not exist in the prediction

predicted color -> INVALID
    prediction contains a pixel that does not exist in the target
~~~

This preserves categorical error information without inventing a continuous distance between ARC colors.

---

## 5. Loss nodes can be solved independently

Every node in the loss tree contains:

~~~python
solution_idx = -1
~~~

where -1 means no gene/program has yet been identified as solving that subproblem.

As GP search discovers explanations, nodes can point back into the gene matrix:

~~~python
tree["shape"]["h"].solution_idx = gene_idx
tree["shape"]["w"].solution_idx = gene_idx
tree["composite"].solution_idx = gene_idx
~~~

This allows the search to reason about partial progress without forcing every candidate program to solve the entire task at once.

For example, one gene may perfectly determine output height while providing no information about output colors. That is still useful information and should constrain the remaining search.

---

## 6. Estimated search process

The current expected process is approximately:

~~~text
1. Load demonstration pairs
        |
        v
2. Initialize GP_Set
        |
        v
3. Store raw input genes
        |
        v
4. Classify exact gene equivalence
   unif / uniq / part
        |
        v
5. Apply candidate operations
        |
        v
6. Append derived genes
        |
        v
7. Test newly created relationships
        |
        v
8. Generate candidate output / output attribute
        |
        v
9. Resolve hierarchical loss tree
        |
        v
10. Record which genes solve which loss nodes
        |
        v
11. Expand promising unresolved branches
        |
        v
12. Backtrack / compose transformations
        |
        v
13. Reach zero unresolved output constraints
        |
        v
14. Apply discovered program to test input
~~~

The important optimization is that the search should not blindly enumerate every possible program to full depth.

The loss hierarchy provides a way to search toward **constraint reduction**.

A transformation is valuable when it explains some previously unresolved portion of the output while remaining consistent across the demonstrations.

The search can therefore move deeply into useful abstractions, backtrack when an interpretation stops constraining the answer, and combine independently discovered solutions.

---

## 7. Genetic Programming interpretation

A candidate program can be viewed as a lineage through the gene matrix:

~~~text
raw grid
   |
grid_dissection
   |
component extraction
   |
attribute selection
   |
transformation
   |
candidate output attribute
~~~

Each derived gene retains its generating operation and source, allowing the solver to reconstruct the program that produced it.

The intended search space is therefore not just a collection of final predictions. It is a graph/tree of increasingly abstract **interpretable transformations**.

This makes GP useful here for two reasons:

1. operations can be composed into symbolic programs;
2. the search can retain partial discoveries that solve only part of the loss tree.

Future optimizations may include grammar restrictions, complexity penalties, equivalence pruning, partition-aware search, expected constraint gain, caching, backtracking, and stochastic/tree-search strategies.

---

## 8. Definition of success

For a training example, the output is solved when:

~~~text
height resolved
AND
width resolved
AND
composite transition matrix contains no off-diagonal error
~~~

Across the task, the desired program must reproduce **every demonstration pair exactly**.

Only after the program satisfies the demonstrations is it applied to the unseen test grid.

This repository intentionally distinguishes:

~~~text
finding a program that fits the examples
~~~

from:

~~~text
finding a simple/reliable program likely to generalize
~~~

The first is exact constraint satisfaction.

The second is the central research problem.

---

## Quick start

~~~bash
git clone https://github.com/Logan-Kelsch/ARC-AGI-2_For_Fun.git
cd ARC-AGI-2_For_Fun

make setup
make data
make test
~~~

**make data** clones the official ARC-AGI-2 data into **data/ARC-AGI-2**.

Useful commands:

~~~bash
make list-tasks SPLIT=training
make inspect TASK=<task_id> SPLIT=training
make evaluate
~~~

---

## Current analytical operations

The experimental GP work currently lives primarily in:

~~~text
notebooks/ops/
├── GP.py
├── eval.py
├── grid_ops.py
└── alpha_ops.py
~~~

Important current pieces include:

~~~python
init_gp_mat(...)
gp_fill_status(...)
grid_dissection(...)
loss_resolution(...)
color_transition_matrix(...)
~~~

The **src/arc_agi2_fun/** package still contains the stable task loading, scoring, evaluation, visualization, and submission infrastructure around the experimental solver.

---

## Repository structure

~~~text
src/arc_agi2_fun/
    data.py
    scoring.py
    programs.py
    search.py
    solver.py
    registry.py
    evaluation.py
    submission.py
    visualize.py
    tensors.py

notebooks/
    ops/
        GP.py
        eval.py
        grid_ops.py
        alpha_ops.py

tests/
docs/
scripts/
~~~

---

## Current research principle

The solver should avoid assuming that ARC colors, objects, or transformations have meaningful continuous distances unless an operation explicitly defines one.

The working philosophy is:

> **Represent exactly, partition exactly, decompose the remaining uncertainty, and search for the smallest consistent program that removes it.**

This is the current estimated process, not a claim that the complete ARC solver has already been implemented.
