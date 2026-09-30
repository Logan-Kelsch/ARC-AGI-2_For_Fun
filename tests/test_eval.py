from notebooks.ops import eval as eval_ops
from notebooks.ops import loss as loss_ops


def test_eval_module_reexports_canonical_loss_api():
    assert eval_ops.LossNode is loss_ops.LossNode
    assert eval_ops.LossTree is loss_ops.LossTree
    assert eval_ops.color_transition_matrix is loss_ops.color_transition_matrix
    assert eval_ops.dimension_transition_matrix is loss_ops.dimension_transition_matrix
    assert eval_ops.loss_resolution is loss_ops.loss_resolution
    assert eval_ops.resolve_loss_tree is loss_ops.resolve_loss_tree
    assert eval_ops.inspect_loss is loss_ops.inspect_loss
