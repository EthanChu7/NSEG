import dgl
import torch
import torch.nn.functional as F
import numpy as np
from GCN.model import GCN, GIN
from explainer.explainer_NSEG import NSEG
from sklearn.metrics import roc_auc_score
from configs.utils import Selector
import argparse
from explainer.utils import collect_evaluations_fid_edge



def arg_parse():
    parser = argparse.ArgumentParser(description="NSEG arguments")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, help="Mutagenicity")
    parser.add_argument("--device", dest="device", type=str, help="cpu or cuda")
    parser.add_argument("--seed", dest="seed", type=int, help="seed")
    parser.add_argument("--model_arc", dest="model_arc", type=str, help="model_arc")
    parser.set_defaults(
        dataset_name="MSRC_21",
        device="cuda",
        seed=0,
        model_arc='GIN'
    )
    return parser.parse_args()


def main():
    # load the arguments
    prog_args = arg_parse()
    dataset_name = prog_args.dataset_name
    device = prog_args.device
    seed = prog_args.seed
    model_arc = prog_args.model_arc
    print(dataset_name, device, seed, model_arc)


    torch.manual_seed(seed)

    # load checkpoint:
    path = "./checkpoint/{}_graph/{}.pth".format(model_arc, dataset_name)
    ckpt = torch.load(path)
    cg_data = ckpt['cg_data']
    model_state = ckpt['model_state']
    # dataset = cg_data['dataset']
    dim_input = cg_data['dim_input']
    dim_hidden = cg_data['dim_hidden']
    num_classes = cg_data['num_classes']
    dataset = dgl.data.LegacyTUDataset(dataset_name)
    graphs = dataset.graph_lists
    labels = dataset.graph_labels
    print("model info: ")
    print("classes: {}, "
          "dim_input: {}, "
          "dim_hidden: {}, "
          "acc_test: {}, "
          "epoch: {}".format(num_classes, dim_input, dim_hidden, cg_data["acc_test"], cg_data["epoch"]))
    print('-----------------------------------------------------------------------------------------------')

    # load GNN model from checkpoint:
    if model_arc == 'GCN':
        model = GCN(dim_input, dim_hidden, num_classes, mode='graph')
    elif model_arc == 'GIN':
        model = GIN(dim_input, dim_hidden, num_classes, mode='graph')

    model.load_state_dict(model_state)

    # load the configuration of NSEG(PNS^{e}) from json:
    config_path = 'configs/{}_edge.json'.format(dataset_name)
    config = Selector(config_path).args
    alpha_e = config.alpha_e
    beta_e = config.beta_e
    alpha_f = config.alpha_f
    beta_f = config.beta_f
    objective = config.objective
    type_ex = config.type_explanation
    num_epochs = config.num_epochs
    lr = config.lr

    print(dataset_name, device, alpha_e, beta_e, alpha_f, beta_f, objective, type_ex, num_epochs, lr)

    # initialize NSEG:
    explainer = NSEG(model=model,
                     num_hops=3,
                     alpha_e=alpha_e,
                     beta_e=beta_e,
                     alpha_f=alpha_f,
                     beta_f=beta_f,
                     num_epochs=num_epochs,
                     objective=objective,
                     type_ex=type_ex,
                     lr=lr,
                     device=device)


    if dataset_name == 'Mutagenicity':
        print('this is Mutagenicity dataset in getting indices')
        object_gt = np.load('dataset/Mutagenicity/Mutagenicity_ground_truths.npy', allow_pickle=True)
        indices_graph = object_gt[0]
        ground_truths = object_gt[1]
        indices_graph = indices_graph[:250]
    else:
        indices_graph = list(range(0, 250))

    fids_plus = []
    fids_minus = []
    fids_plus_dis = []
    fids_minus_dis = []
    explanations = []

    thres = 0.5
    for idx in range(len(indices_graph)):
        # explain graph:
        graph = graphs[indices_graph[idx]].to(device)
        feat = graph.ndata['feat'].to(device)
        mask_edge, edges_ids = explainer.explain_graph(indices_graph[idx], graph, feat)
        print(mask_edge.shape)
        print(torch.sum((mask_edge>=0.5).int()))

        explanations.append([mask_edge.detach().cpu().numpy(), edges_ids])

        # evaluate:
        model.eval()
        f_plus, f_minus, f_plus_dis, f_minus_dis = collect_evaluations_fid_edge(model=model,
                                                                                graph=graph,
                                                                                features=feat,
                                                                                mask_edge=mask_edge,
                                                                                node_idx=0,
                                                                                thres=thres,
                                                                                is_submask=False)


        fids_plus.append(f_plus)
        fids_minus.append(f_minus)
        fids_plus_dis.append(f_plus_dis)
        fids_minus_dis.append(f_minus_dis)

        print("fid_plus: {}, fid_minus: {}, fid_plus_dis: {}, fid_minus_dis: {}".format(f_plus,
                                                                                        f_minus,
                                                                                        f_plus_dis,
                                                                                        f_minus_dis))

    macro_fid_plus = np.mean(fids_plus)
    macro_fid_minus = np.mean(fids_minus)
    macro_fid_plus_dis = np.mean(fids_plus_dis)
    macro_fid_minus_dis = np.mean(fids_minus_dis)
    char_score = (2 * macro_fid_plus * (1 - macro_fid_minus)) / (macro_fid_plus + 1 - macro_fid_minus)
    char_score_dis = (2 * macro_fid_plus_dis * (1 - macro_fid_minus_dis)) / (
            macro_fid_plus_dis + 1 - macro_fid_minus_dis)

    print("macro_fid_plus: {}, macro_fid_minus: {}, charact: {}".format(macro_fid_plus, macro_fid_minus, char_score))
    print("threshold: {}, macro_fid_plus_dis: {}, macro_fid_minus_dis: {}, charact_dis: {}".format(thres,
                                                                                                   macro_fid_plus_dis,
                                                                                                   macro_fid_minus_dis,
                                                                                                   char_score_dis))
    path = './res/{}/NSEG_explanation_{}_{}_{}_{}_{}_{}_{}.npy'.format(dataset_name, alpha_e, beta_e, objective, type_ex,
                                                                       num_epochs, seed, model_arc)
    print('path:', path)
    np.save(path, np.array(explanations))


if __name__ == "__main__":
    main()

