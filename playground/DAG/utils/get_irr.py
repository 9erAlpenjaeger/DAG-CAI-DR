import networkx as nx

def get_irrelevant(nx_G):
    # calculate the transitive closure
    irrelevant_pair = []
    tc_nx_G = nx.transitive_closure_dag(nx_G)
    for n in nx_G.nodes():
        nx_G.nodes[n]['irr'] = 0
        nx_G.nodes[n]['irr_adj'] = [0] * nx_G.number_of_nodes()
        nx_G.nodes[n]['irr_adj'][n] = 1
    for n in tc_nx_G.nodes():
        for m in tc_nx_G.nodes():
            if n != m:
                if n not in tc_nx_G[m] and m not in tc_nx_G[n]:
                    nx_G.nodes[n]['irr_adj'][m] = 1
                    if m > n:
                        ''' 保证irr_pair不重复'''
                        irrelevant_pair.append([m, n])
                    nx_G.nodes[n]['irr'] = 1
                    nx_G.nodes[m]['irr'] = 1
    return irrelevant_pair

def get_irrelevant_jssp(nx_G):
    machine_operation_map = dict()
    for node in nx_G.nodes():
        machine = nx_G.nodes[node]['machine']
        if machine in machine_operation_map:
            machine_operation_map[machine].append(node)
        else:
            machine_operation_map[machine] = [node]
    
    irrelevant_pair = []
    for n in nx_G.nodes():
        nx_G.nodes[n]['irr'] = 0
        nx_G.nodes[n]['irr_adj'] = [0] * nx_G.number_of_nodes()
        machine = nx_G.nodes[n]['machine']
        operation_map = machine_operation_map[machine]
        irrelevant_pair.append([n, n])
        for n_wm in operation_map:
            nx_G.nodes[n]['irr_adj'][n_wm] = 1
            nx_G.nodes[n]['irr'] = 1
            if n > n_wm:
                irrelevant_pair.append([n, n_wm])

    return irrelevant_pair