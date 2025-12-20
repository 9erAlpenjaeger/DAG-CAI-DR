import os
os.environ['PYTORCH_GEOMETRIC_USE_OPT'] = '0'  
import numpy as np

from scipy.sparse.csgraph import shortest_path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import softmax
import math

from torchvision import ops


def collect_demo(demo_list):
    # demo_list: E * (B * N)
    demo_action = np.stack(demo_list, axis = 1)
    # demo_action: (B * E * N)
    return demo_action

class DemonstrationBuffer:
    def __init__(self, data: np.ndarray, max_episode = 256):
        self.data = np.array(data)
        self.size = len(self.data)
        self.available_indices = set(range(self.size))
        self.episode_counter = 0
        self.max_episode = max_episode

    def draw(self):
        if not self.available_indices:
            self.available_indices = set(range(self.size))
            self.episode_counter += 1

        idx = np.random.choice(list(self.available_indices))
        self.available_indices.remove(idx)
        return self.data[idx] 


###################################################################################################################

def cat_list(list_input):
    if len(list_input) > 1:
        list_output = torch.stack(list_input)
    else:
        list_output = list_input[0].unsqueeze(0)
    return list_output

####################################################################################################################

def sample_gumbel(shape, device, eps=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

class GumbelSort(torch.nn.Module):
    def __init__(self):
        super(GumbelSort, self).__init__()
        
    def forward(self, logits, is_evaluate = False):
        if is_evaluate:
            gumbel_logits = logits
        else:
            gumbel_logits = logits + sample_gumbel(shape = logits.size(), device = logits.device)
        rank_mask = (gumbel_logits[:, None] >= gumbel_logits[None, :])#.float()  # for rew_1 and rew2
        #sf = torch.exp(gumbel_logits)
        logits_expand_r = logits.unsqueeze(0).expand(logits.shape[0], -1)
        logits_expand_c = logits.unsqueeze(1).expand(-1, logits.shape[0])
        rd = (torch.exp(logits_expand_r - logits_expand_c)).masked_fill(~rank_mask, 0)
        log_prob = 0 - torch.log(rd.sum(dim=1) + 1e-10)
        
        #sf = torch.exp(logits - logits.max(dim = -1, keepdim = True).values) # 可能出现趋近0的情况
        #rd = rank_mask @ sf + 1e-10
        #log_prob = torch.log(sf / rd)
        return log_prob, gumbel_logits

class GumbelTopoSortInTuples(torch.nn.Module):
    def __init__(self):
        super(GumbelTopoSortInTuples, self).__init__()    
        
    def forward(self, logits, valid_irr_adj, is_evaluate = False, given_logits = None):
        if is_evaluate:
            gumbel_logits = given_logits
        else:
            gumbel_noise = sample_gumbel(shape = logits.size(), device = logits.device)
            gumbel_logits = logits + gumbel_noise
        #sorted_logits, indices = torch.sort(gumbel_logits, descending=True )
        node_num = int(gumbel_logits.shape[0])
        valid_irr_adj = valid_irr_adj[:node_num,:node_num]

        rank_mask = (gumbel_logits[:, None] >= gumbel_logits[None, :]).float()  # for rew_1 and rew2
        integrated_mask = rank_mask #* valid_irr_adj 
        #sf = torch.exp(logits)
        unselected_mask = torch.ones(node_num, dtype=torch.bool, device = logits.device)
        '''for i in range(node_num):
            #zerodin_mask = (adj_mat.sum(dim = 0) == 0) * unselected_mask
            # 因为是升序排序所以取反..是否正确？
            masked_gumbel_logits = gumbel_logits.masked_fill(~unselected_mask,float('-inf')) #去除已选的和入度不为0的，留下入度为0的
            max_node_idx = torch.argmax(masked_gumbel_logits)
            unselected_mask[max_node_idx] = False'''
        ''' 防止指数溢出'''        
        logits_expand_r = logits.unsqueeze(0).expand(logits.shape[0], -1)
        logits_expand_c = logits.unsqueeze(1).expand(-1, logits.shape[0])
        rd = (torch.exp(logits_expand_r - logits_expand_c)).masked_fill(~integrated_mask, 0)
        log_prob = 0 - torch.log(rd.sum(dim=1) + 1e-10)
        return log_prob, gumbel_logits

class GumbelTopoSortAdj(torch.nn.Module):
    def __init__(self):
        super(GumbelTopoSortAdj, self).__init__()    
        
    def forward(self, logits, adj_mat, is_evaluate = False, given_logits = None): 
        node_num = int(logits.shape[0])
        adj_mat = (adj_mat[:node_num, :node_num]).clone()
        if is_evaluate:
            gumbel_logits = given_logits[:node_num]
        else:
            gumbel_noise = sample_gumbel(shape = logits.size(), device = logits.device)
            gumbel_logits = logits + gumbel_noise
        
        integrated_mask = torch.zeros((node_num, node_num), dtype=torch.bool, device = logits.device) 
        unselected_mask = torch.ones(node_num, dtype=torch.bool, device = logits.device)
        for i in range(node_num):
            #zerodin_mask = (adj_mat.sum(dim = 0) == 0) * unselected_mask
            zerodin_mask = (adj_mat.sum(dim = 1) == 0) * unselected_mask # 
            masked_gumbel_logits = gumbel_logits.masked_fill(~zerodin_mask,float('-inf')) 
            max_node_idx = torch.argmax(masked_gumbel_logits)
            unselected_mask[max_node_idx] = False
            adj_mat[max_node_idx, : ] = 0
            adj_mat[:, max_node_idx ] = 0
            integrated_mask[max_node_idx] = zerodin_mask
            
        #sf = torch.exp(logits)
        logits_expand_r = logits.unsqueeze(0).expand(logits.shape[0], -1)
        logits_expand_c = logits.unsqueeze(1).expand(-1, logits.shape[0])
        rd = torch.exp(logits_expand_r - logits_expand_c)*integrated_mask
        log_prob = 0 - torch.log(rd.sum(dim=1) + 1e-10)

        return log_prob, gumbel_logits

####################################################################################################################

class GATConvWithLogits(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.att = torch.nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att.data)
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin(x)
        edge_index = edge_index.to(torch.long)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index, ptr, size_i):
        
        index = index.to(torch.long)
        alpha_raw = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) 
        self._raw_attention = alpha_raw #.detach()
        alpha = softmax(alpha_raw, index)

        return x_j * alpha.view(-1, 1)

#####################################################################################################################################
def init_params(module, n_layers):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    
def derive_prob_over_selected_set(adj_mat_weight, zerodin_mask, unselected_mask):
    row_mask = (~unselected_mask) #.view(-1, 1) # Schduleds
    col_mask = zerodin_mask #.view(1, -1)  # candidates

    row_indices = torch.nonzero(row_mask, as_tuple=True)[0]  # shape: (num_rows,)
    col_indices = torch.nonzero(col_mask, as_tuple=True)[0]  # shape: (num_cols,)
    row_grid, col_grid = torch.meshgrid(row_indices, col_indices, indexing='ij') 
    #masked_adj_mat_weight = adj_mat_weight[row_mask][:, col_mask]
    masked_adj_mat_weight = adj_mat_weight[row_grid, col_grid]

    softmax_result = F.softmax(masked_adj_mat_weight.view(-1), dim=0).view_as(masked_adj_mat_weight)

    final_result = torch.zeros_like(adj_mat_weight, dtype=torch.float32, device = adj_mat_weight.device)
    #final_result[row_mask][:, col_mask] = softmax_result
    final_result[row_grid, col_grid] = softmax_result

    probs = final_result
    return probs