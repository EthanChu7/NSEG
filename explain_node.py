import torch
import numpy as np
from configs.utils import Selector
import argparse

from explainer.explainer_NSEG import NSEG
from explainer.utils import collect_evaluations_fid_edge, collect_evaluations_acc_edge
from GCN.model import GCN, GIN


def arg_parse():
    parser = argparse.ArgumentParser(description="NSEG arguments")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, help="BA-Shapes, Tree-Cycles, or Tree-Grid")
    parser.add_argument("--device", dest="device", type=str, help="cpu or cuda")
    parser.add_argument("--seed", dest="seed", type=int, help="seed")
    parser.add_argument("--model_arc", dest="model_arc", type=str, help="model_arc")
    parser.set_defaults(
        dataset_name="BA-Shapes",
        device="cpu",
        seed=0,
        model_arc='GIN'
    )
    return parser.parse_args()


def main():
    # load the arguments:
    prog_args = arg_parse()
    dataset_name = prog_args.dataset_name
    device = prog_args.device
    seed = prog_args.seed
    model_arc = prog_args.model_arc
    print(dataset_name, device, seed, model_arc)

    torch.manual_seed(seed)

    # the instance indices in motifs
    if dataset_name == 'BA-Shapes':
        nodes = list(range(300, 700))
        k_edge = 6
    elif dataset_name == 'Tree-Cycles':
        nodes = list(range(511, 871))
        k_edge = 6
    elif dataset_name == 'Tree-Grid':
        nodes = list(range(511, 1231))
        k_edge = 12

    # load checkpoint:
    # path = "./checkpoint/GCN/" + dataset_name + '.pth'
    path = './checkpoint/{}/{}.pth'.format(model_arc, dataset_name)
    print('path:', path)
    ckpt = torch.load(path)
    cg_data = ckpt["cg_data"]
    model_state = ckpt["model_state"]
    graph = cg_data["graph"].to(device)
    features = cg_data["features"].to(device)
    labels = cg_data["labels"].to(device)
    dim_input = cg_data["dim_input"]
    dim_hidden = cg_data["dim_hidden"]
    num_classes = cg_data["num_classes"]
    num_layers = cg_data["num_layers"]
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print("# nodes: {}, # edges: {}".format(num_nodes, num_edges / 2))
    print("dataset: {}".format(dataset_name))
    print("model info: ")
    print("classes: {}, "
          "dim_input: {}, "
          "dim_hidden: {}, "
          "acc_test: {}, "
          "epoch: {}, "
          "time: {} s".format(num_classes, dim_input, dim_hidden, cg_data["acc_test"],
                              cg_data["epoch"], cg_data["time"]))
    print('-----------------------------------------------------------------------------------------------')

    # load GNN model from checkpoint:
    if model_arc == 'GCN':
        model = GCN(dim_input, dim_hidden, num_classes, num_layers=num_layers)
    elif model_arc == 'GIN':
        model = GIN(dim_input, dim_hidden, num_classes, num_layers=num_layers)
    model.load_state_dict(model_state)
    model.eval()

    # load configurations of NSEG(PNS^{e}) from json:
    config_path = 'configs/{}.json'.format(dataset_name)
    config = Selector(config_path).args
    print(config)
    alpha_e = config.alpha_e
    beta_e = config.beta_e
    alpha_f = config.alpha_f
    beta_f = config.beta_f
    objective = config.objective
    type_ex = config.type_explanation
    num_epochs = config.num_epochs
    lr = config.lr

    # initialize NSEG:
    explainer = NSEG(model=model,
                     num_hops=num_layers,
                     alpha_e=alpha_e,
                     beta_e=beta_e,
                     alpha_f=alpha_f,
                     beta_f=beta_f,
                     num_epochs=num_epochs,
                     objective=objective,
                     type_ex=type_ex,
                     device=device,
                     lr=lr)

    aucs = []
    fids_plus = []
    fids_minus = []
    fids_plus_dis = []
    fids_minus_dis = []
    recalls_k = []
    explanations = []
    thres = 0.5
    for node in nodes:
        # explain node:
        mask_edge_sigmoid, edge_ids = explainer.explain_node(node, graph, features)

        explanations.append([mask_edge_sigmoid.detach().cpu().numpy(), edge_ids.detach().cpu().numpy()])

        # evaluate:
        model.eval()
        recall_k, auc = collect_evaluations_acc_edge(mask_edge=mask_edge_sigmoid,
                                                     edge_ids=edge_ids,
                                                     node_idx=node,
                                                     dataset=dataset_name,
                                                     graph=graph,
                                                     k=k_edge,
                                                     is_submask=False)

        f_plus, f_minus, f_plus_dis, f_minus_dis = collect_evaluations_fid_edge(model=model,
                                                                                graph=graph,
                                                                                features=features,
                                                                                mask_edge=mask_edge_sigmoid,
                                                                                node_idx=node,
                                                                                thres=thres,
                                                                                is_submask=False)

        if auc != -1:
            aucs.append(auc)
        recalls_k.append(recall_k)
        fids_plus.append(f_plus)
        fids_minus.append(f_minus)
        fids_plus_dis.append(f_plus_dis)
        fids_minus_dis.append(f_minus_dis)

        print('node_idx: {}'.format(node))
        print("auc: {}, recall@{}: {}, fid_plus: {}, fid_minus: {}, fid_plus_dis: {}, fid_minus_dis: {}".format(auc,
                                                                                                                k_edge,
                                                                                                                recall_k,
                                                                                                                f_plus,
                                                                                                                f_minus,
                                                                                                                f_plus_dis,
                                                                                                                f_minus_dis))

    macro_recall_k = np.mean(recalls_k)
    macro_auc = np.mean(aucs)
    macro_fid_plus = np.mean(fids_plus)
    macro_fid_minus = np.mean(fids_minus)
    macro_fid_plus_dis = np.mean(fids_plus_dis)
    macro_fid_minus_dis = np.mean(fids_minus_dis)
    char_score = (2 * macro_fid_plus * (1 - macro_fid_minus)) / (macro_fid_plus + 1 - macro_fid_minus)
    char_score_dis = (2 * macro_fid_plus_dis * (1 - macro_fid_minus_dis)) / (
                macro_fid_plus_dis + 1 - macro_fid_minus_dis)

    print("macro_auc: {}, macro_recall@{}: {}".format(macro_auc, k_edge, macro_recall_k))
    print("macro_fid_plus: {}, macro_fid_minus: {}, charact: {}".format(macro_fid_plus, macro_fid_minus, char_score))
    print("threshold: {}, macro_fid_plus_dis: {}, macro_fid_minus_dis: {}, charact_dis: {}".format(thres,
                                                                                                   macro_fid_plus_dis,
                                                                                                   macro_fid_minus_dis,
                                                                                                   char_score_dis))
    path = './res/{}/NSEG_explanation_{}_{}_{}_{}_{}_{}_{}.npy'.format(dataset_name, alpha_e, beta_e, objective, type_ex,
                                                                       num_epochs, seed, model_arc)
    print('save path:', path)
    np.save(path, np.array(explanations))

if __name__ == "__main__":
    main()
