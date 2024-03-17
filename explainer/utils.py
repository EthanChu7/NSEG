import dgl
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import math




def get_ground_truth_syn1(node):
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    return ground_truth

def get_ground_truth_syn3(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]
    return ground_truth

def get_ground_truth_syn4(node):
    buff = node - 1
    base = [0, 1, 2, 3, 4, 5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]
    return ground_truth

def get_ground_truth_syn5(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]
    return ground_truth

# def get_ground_truth_syn5(self, node):
#     base = [0, 1, 2, 3]
#     buff = node - 2
#     ground_truth = []
#     offset = buff % 4
#     ground_truth = [buff - offset + val + 2 for val in base]
#     return ground_truth

def get_ground_truth(node, dataset):

    gt = []
    if dataset == 'BA-Shapes':
        gt = get_ground_truth_syn1(node)# correct
    elif dataset == 'syn2':
        gt = get_ground_truth_syn1(node)  # correct
    elif dataset == 'syn3':
        gt = get_ground_truth_syn3(node)  # correct
    elif dataset == 'Tree-Cycles':
        gt = get_ground_truth_syn4(node)  # correct
    elif dataset == 'Tree-Grid':
        gt = get_ground_truth_syn5(node)  # correct
    elif dataset == 'syn6':
        gt = get_ground_truth_syn1(node)  # correct
    return gt

def ground_truth(node, dataset, graph):
    node_gt = get_ground_truth(node, dataset)
    sg = dgl.node_subgraph(graph, node_gt)

    edge_gt = list(sg.edata[dgl.EID].detach().cpu().numpy())

    edges = graph.edges()
    for edge_id in edge_gt:
        if edges[0][edge_id] == edges[1][edge_id]:
            edge_gt.remove(edge_id)

    # print(node_gt)
    # # graph = dgl.to_networkx(sg)
    # # nx.draw(graph)
    # # plt.show()
    return edge_gt, node_gt

def accuracy_precision(edge_explanation, node_explanation, idx_node, dataset, graph, is_node_involed=False):
    gt_positive_edge = 0.
    true_positive_edge = 0.
    pred_positive_edge = 0.

    edge_gt, node_gt = ground_truth(idx_node, dataset, graph)

    gt_positive_edge = gt_positive_edge + len(edge_gt)
    pred_positive_edge = pred_positive_edge + len(edge_explanation)

    for ex_edge in edge_explanation:
        if ex_edge in edge_gt:
            true_positive_edge = true_positive_edge + 1

    if gt_positive_edge == 0 or pred_positive_edge == 0:
        return 0, 0

    accuracy_edge = true_positive_edge / gt_positive_edge
    precision_edge = true_positive_edge / pred_positive_edge

    # print(edge_gt)

    if is_node_involed:
        gt_positive_node = 0.
        true_positive_node = 0.
        pred_positive_node = 0.

        gt_positive_node = gt_positive_node + len(node_gt)
        pred_positive_node = pred_positive_node + len(node_explanation)

        for ex_node in node_explanation:
            if ex_node in node_gt:
                true_positive_node = true_positive_node + 1

        accuracy_node = true_positive_node / gt_positive_node
        precision_node = true_positive_node / pred_positive_node

        return accuracy_edge, precision_edge, accuracy_node, precision_node
    else:
        return accuracy_edge, precision_edge





def get_auc_mask(mask_pred, edge_ids, idx_node, dataset, graph):

    gts, _ = ground_truth(idx_node, dataset, graph)
    new_gts = []
    for i in range(edge_ids.shape[0]):
        if edge_ids[i] in gts:
            new_gts.append(i)

    mask_true = np.zeros(shape=mask_pred.shape)
    for new_gt in new_gts:
        mask_true[new_gt] = 1

    if mask_true.sum() == mask_true.shape[0]:
        auc = 1
    else:
        auc = roc_auc_score(mask_true, mask_pred)
    return auc

def get_auc_mask_node(mask_pred, node_ids, idx_node, dataset, graph):
    _, gts = ground_truth(idx_node, dataset, graph)
    new_gts = []
    for i in range(node_ids.shape[0]):
        if node_ids[i] in gts:
            new_gts.append(i)

    mask_true = np.zeros(shape=mask_pred.shape)
    for new_gt in new_gts:
        mask_true[new_gt] = 1

    if mask_true.sum() == mask_true.shape[0]:
        auc = -1
    else:
        auc = roc_auc_score(mask_true, mask_pred)
    return auc


def get_recall_k_edge(edge_explanation, idx_node, dataset, graph):
    gt_positive_edge = 0.
    true_positive_edge = 0.
    pred_positive_edge = 0.

    edge_gt, _ = ground_truth(idx_node, dataset, graph)

    gt_positive_edge = gt_positive_edge + len(edge_gt)
    pred_positive_edge = pred_positive_edge + len(edge_explanation)

    for ex_edge in edge_explanation:
        if ex_edge in edge_gt:
            true_positive_edge = true_positive_edge + 1

    if gt_positive_edge == 0 or pred_positive_edge == 0:
        return 0

    recall_k_edge = true_positive_edge / gt_positive_edge

    return recall_k_edge

def get_opposite_edge_id(edge_id, graph):
    edges = graph.edges()
    for i in range(len(edges[0])):
        if edges[0][i] == edges[1][edge_id] and edges[1][i] == edges[0][edge_id]:
            return i

def get_top_k_edges(graph, mask_edge_sigmoid, edge_ids, thres_edge_num=20):
    if isinstance(mask_edge_sigmoid, torch.Tensor):
        mask_edge_sigmoid = mask_edge_sigmoid.clone().detach().cpu().numpy()

    idx_sorted_edge = np.argsort(mask_edge_sigmoid)
    idx_sorted_edge = idx_sorted_edge[::-1]

    edge_indices = np.array(edge_ids)[idx_sorted_edge]

    # print(edge_indices)

    edge_explanation = []
    for edge_id in edge_indices:
        if len(edge_explanation) >= 2 * thres_edge_num:  # directed edge, x2
            break
        if edge_id not in edge_explanation:
            op_edge_id = get_opposite_edge_id(edge_id, graph)
            if op_edge_id == edge_id:  # self-loop
                continue
            # print(edge_id, op_edge_id)
            edge_explanation.append(edge_id)
            edge_explanation.append(op_edge_id)

    subgraph = dgl.edge_subgraph(graph, edge_indices)

    # print('g', edge_explanation)
    return edge_explanation, subgraph

def get_logit_with_mask(model, graph, features, mask=None, apply_softmax=False):
    model.eval()
    with torch.no_grad():
        logit = model(graph, features, mask)
    if apply_softmax:
        logit = F.softmax(logit, dim=1)
    return logit

def collect_evaluations_acc_edge(mask_edge, edge_ids, node_idx, dataset, graph, k, is_submask=True):
    """

    :param mask_edge: the explanation mask, if is_submask=True, the mase is the sub_mask, else the mask is the whole mask
    :param edge_ids: the indices of edges
    :param node_idx: the
    :param dataset:
    :param graph:
    :param k:
    :param threshold:
    :param is_submask:
    :return:
    """
    if not is_submask:
        mask_edge = mask_edge[edge_ids]
    top_k_explanations, _ = get_top_k_edges(graph, mask_edge.detach().cpu(), edge_ids.detach().cpu(), k)
    recall_k = get_recall_k_edge(top_k_explanations, node_idx, dataset, graph)
    auc = get_auc_mask(mask_edge.detach().cpu(), edge_ids, node_idx, dataset, graph)

    return recall_k, auc


def collect_evaluations_fid_edge(model, graph, features, mask_edge, node_idx=0, thres=0.5, is_submask=False):
    # for graph classification task node_idx should be set to 0 and is_submask should be set to False

    logit_ori = get_logit_with_mask(model=model, graph=graph, features=features, mask=None)[node_idx]
    pred_ori = torch.argmax(logit_ori)

    if is_submask:
        sg, inverse_indices = dgl.khop_in_subgraph(graph, node_idx, 3)
        features = features[sg.ndata[dgl.NID]]
    else:
        sg = graph
        inverse_indices = node_idx

    logit_plus = get_logit_with_mask(model=model, graph=sg, features=features, mask=(1.-mask_edge))[inverse_indices]
    logit_minus = get_logit_with_mask(model=model, graph=sg, features=features, mask=mask_edge)[inverse_indices]

    pred_plus = torch.argmax(logit_plus)
    pred_minus = torch.argmax(logit_minus)

    if pred_plus == pred_ori:
        fid_plus = 0.
    else:
        fid_plus = 1.

    if pred_minus == pred_ori:
        fid_minus = 0.
    else:
        fid_minus = 1.

    mask_discrete = (mask_edge >= thres).float()

    logit_plus_dis = get_logit_with_mask(model=model, graph=sg, features=features, mask=(1. - mask_discrete))[inverse_indices]
    logit_minus_dis = get_logit_with_mask(model=model, graph=sg, features=features, mask=mask_discrete)[inverse_indices]

    pred_plus_dis = torch.argmax(logit_plus_dis)
    pred_minus_dis = torch.argmax(logit_minus_dis)

    if pred_plus_dis == pred_ori:
        fid_plus_dis = 0.
    else:
        fid_plus_dis = 1.

    if pred_minus_dis == pred_ori:
        fid_minus_dis = 0.
    else:
        fid_minus_dis = 1.

    return fid_plus, fid_minus, fid_plus_dis, fid_minus_dis