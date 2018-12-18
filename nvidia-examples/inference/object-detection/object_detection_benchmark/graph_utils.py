import tensorflow as tf

def remove_node(graph_def, node):
    for n in graph_def.node:
        if node.name in n.input:
            n.input.remove(node.name)
        ctrl_name = '^' + node.name
        if ctrl_name in n.input:
            n.input.remove(ctrl_name)
    graph_def.node.remove(node)

def remove_op(graph_def, op_name):
    matches = [node for node in graph_def.node if node.op == op_name]
    for match in matches:
        remove_node(graph_def, match)

def force_nms_cpu(frozen_graph):
    for node in frozen_graph.node:
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    return frozen_graph

def remove_assert(frozen_graph):
    remove_op(frozen_graph, 'Assert')
    return frozen_graph
