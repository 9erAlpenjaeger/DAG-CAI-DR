import numpy
import torch
#from torch_geometric.utils.convert import from_networkx

def topological_batching(DiG):
    temp_indegrees = {}
    temp_outdegrees= {}
    for n in DiG.nodes():
        temp_indegrees[n] = DiG.in_degree(n)
        temp_outdegrees[n]= DiG.out_degree(n)

def get_pyg_from_job(job):
    DiG = job.nx_G
    for node in DiG.nodes():
        task = DiG.nodes[node]['task']
        task_config = task.task_config
        task_instance = task.task_instances[0]
        DiG.nodes[node]['x']

    pyg = from_networkx(DiG)
    add_order_info_01(pyg)
    return pyg

''' see github.com/vthost/DAGNN/src/utils_dag.py'''
''' DAGNN/src/utils_dag.py and DAGNN/ogbg_code/src/utils_dag.py are the same'''
# see https://github.com/unbounce/pytorch-tree-lstm/blob/66f29a44e98c7332661b57d22501107bcb193f90/treelstm/util.py#L8
# assume nodes consecutively named starting at 0
#
def top_sort(edge_index, graph_size, device):
    node_ids = torch.arange(graph_size, dtype=torch.int64, device=device)
    node_order = torch.zeros(graph_size, dtype=torch.int64, device=device)
    unevaluated_nodes = torch.ones(graph_size, dtype=torch.bool, device=device)
    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]
    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]
        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~torch.isin(node_ids, unready_children)
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False
        n += 1
    return node_order

# to be able to use pyg's batch split everything into 1-dim tensors
def add_order_info_01(graph, device):
    l0 = top_sort(graph.edge_index, graph.num_nodes, device)
    ei2 = torch.stack((graph.edge_index[1], graph.edge_index[0]), dim=0)
    l1 = top_sort(ei2, graph.num_nodes, device)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)]).to(device)
    graph.__setattr__("_bi_layer_idx0", l0)
    graph.__setattr__("_bi_layer_index0", ns)
    graph.__setattr__("_bi_layer_idx1", l1)
    graph.__setattr__("_bi_layer_index1", ns)
    #assert_order(graph.edge_index, l0, ns)
    #assert_order(ei2, l1, ns)

def assert_order(edge_index, o, ns):
    # already processed
    proc = []
    for i in range(max(o)+1):
        # nodes in position i in order
        l = o == i
        l = ns[l].tolist()
        for n in l:
            # predecessors
            ps = edge_index[0][edge_index[1] == n].tolist()
            for p in ps:
                assert p in proc
        proc += l

def add_order_info(graph):
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    layers = torch.stack([top_sort(graph.edge_index, graph.num_nodes), ns], dim=0)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    layers2 = torch.stack([top_sort(ei2, graph.num_nodes), ns], dim=0)
    graph.__setattr__("bi_layer_index", torch.stack([layers, layers2], dim=0))
