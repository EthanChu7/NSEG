import math
from tqdm import tqdm
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class NSEG:
    def __init__(self,
                 model,
                 num_hops,
                 mode='node',
                 device='cpu',
                 num_epochs=1000,
                 lr=0.01,
                 alpha_e=0.05,
                 beta_e=1,
                 alpha_f=0.05,
                 beta_f=1,
                 objective='pns',
                 type_ex='e'):

        self.type_ex = type_ex
        self.objective = objective
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_hops = num_hops

        self.mode = mode
        self.device = device
        print(self.device)
        self.num_epochs = num_epochs
        self.lr = lr

        self.alpha_e = alpha_e
        self.beta_e = beta_e
        self.alpha_f = alpha_f
        self.beta_f = beta_f

        print(self.device)

    def explain_node(self, index_node, graph, features, features_cf=None):
        print('explaining node {}...'.format(index_node))
        self.model.eval()
        sg, _ = dgl.khop_in_subgraph(graph, index_node, self.num_hops)
        edges_ids = sg.edata[dgl.EID]
        nodes_ids = sg.ndata[dgl.NID]

        explainer_module = ExplainerModule(graph=graph,
                                           model=self.model,
                                           features_ori=features,
                                           idx_node=index_node,
                                           device=self.device,
                                           alpha_e=self.alpha_e,
                                           beta_e=self.beta_e,
                                           alpha_f=self.alpha_f,
                                           beta_f=self.beta_f,
                                           objective=self.objective,
                                           type_ex=self.type_ex
                                           ).to(self.device)

        if self.type_ex == 'e':
            optimizer = torch.optim.Adam([explainer_module.mask_edge], lr=self.lr)
        elif self.type_ex == 'f':
            optimizer = torch.optim.Adam([explainer_module.mask_node], lr=self.lr)
        elif self.type_ex == 'ef':
            optimizer = torch.optim.Adam([explainer_module.mask_edge, explainer_module.mask_node], lr=self.lr)
        pbar = tqdm(total=self.num_epochs)
        pbar.set_description('Explaining node {}'.format(index_node))

        explainer_module.train()
        for epoch in range(self.num_epochs):
            if self.type_ex == 'e':
                pr_edge_suff, pr_edge_nec, mask_edge_sigmoid = explainer_module()
                loss = explainer_module.loss_e_or_f(pr_edge_suff, pr_edge_nec, mask_edge_sigmoid)

            elif self.type_ex == 'f':
                pr_node_suff, pr_node_nec, mask_node_sigmoid = explainer_module(features_cf)
                loss = explainer_module.loss_e_or_f(pr_node_suff, pr_node_nec, mask_node_sigmoid)

            elif self.type_ex == 'ef':
                pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid = explainer_module(features_cf)
                loss = explainer_module.loss_ef(pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

        if self.type_ex == 'e':
            return mask_edge_sigmoid, edges_ids
        elif self.type_ex == 'f':
            return mask_node_sigmoid, nodes_ids
        elif self.type_ex == 'ef':
            return mask_edge_sigmoid, edges_ids, mask_node_sigmoid, nodes_ids

    def explain_graph(self, index, graph, features, features_cf=None):
        print('explaining graph {}...'.format(index))

        self.model.eval()
        num_edges = graph.number_of_edges()
        edges_ids = []
        for edge in range(num_edges):
            edges_ids.append(edge)

        num_nodes = graph.number_of_nodes()
        nodes_ids = []
        for node in range(num_nodes):
            nodes_ids.append(node)

        explainer_module = ExplainerModuleGraph(graph=graph,
                                                model=self.model,
                                                features_ori=features,
                                                device=self.device,
                                                alpha_e=self.alpha_e,
                                                beta_e=self.beta_e,
                                                alpha_f=self.alpha_f,
                                                beta_f=self.beta_f,
                                                objective=self.objective,
                                                type_ex=self.type_ex
                                                ).to(self.device)

        if self.type_ex == 'e':
            optimizer = torch.optim.Adam([explainer_module.mask_edge], lr=self.lr)
            print('extypee')
        elif self.type_ex == 'f':
            optimizer = torch.optim.Adam([explainer_module.mask_node], lr=self.lr)
        elif self.type_ex == 'ef':
            optimizer = torch.optim.Adam([explainer_module.mask_edge, explainer_module.mask_node], lr=self.lr)

        pbar = tqdm(total=self.num_epochs)
        pbar.set_description('Explaining graph {}'.format(index))

        explainer_module.train()
        for epoch in range(self.num_epochs):
            if self.type_ex == 'e':
                pr_edge_suff, pr_edge_nec, mask_edge_sigmoid = explainer_module()
                loss = explainer_module.loss_e_or_f(pr_edge_suff, pr_edge_nec, mask_edge_sigmoid)
            elif self.type_ex == 'f':
                pr_node_suff, pr_node_nec, mask_node_sigmoid = explainer_module(features_cf)
                loss = explainer_module.loss_e_or_f(pr_node_suff, pr_node_nec, mask_node_sigmoid)
            elif self.type_ex == 'ef':
                pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid = explainer_module(features_cf)
                loss = explainer_module.loss_ef(pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

        pbar.close()

        if self.type_ex == 'e':
            return mask_edge_sigmoid, edges_ids
        elif self.type_ex == 'f':
            return mask_node_sigmoid, nodes_ids
        elif self.type_ex == 'ef':
            return mask_edge_sigmoid, edges_ids, mask_node_sigmoid, nodes_ids



class ExplainerModule(nn.Module):
    def __init__(self, graph, model, features_ori, idx_node, device, alpha_e, beta_e, alpha_f, beta_f, objective='pns', type_ex='e'):
        """

        :param graph: dgl graph
        :param model: fixed GNN model, dgl
        :param features_ori: original features tensor
        """
        super(ExplainerModule, self).__init__()
        self.objective = objective
        self.type_ex = type_ex
        self.graph = graph.to(device)
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.features_ori = features_ori.to(device)
        self.idx_node = idx_node
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.mask_node = nn.Parameter(torch.FloatTensor(self.num_nodes, ).to(device))
        self.mask_edge = nn.Parameter(torch.FloatTensor(self.num_edges, ).to(device))
        self.reset_parameter()
        self.label, self.pr_ori = self.original_prediction()
        self.device = device

        self.alpha_e = alpha_e
        self.beta_e = beta_e
        self.alpha_f = alpha_f
        self.beta_f = beta_f

    def reset_parameter(self, type_init='uniform'):
        if type_init == 'uniform':
            stdv_node = 1. / math.sqrt(self.mask_node.shape[0])
            stdv_edge = 1. / math.sqrt(self.mask_edge.shape[0])
            self.mask_node.data.uniform_(-stdv_node, stdv_node)
            self.mask_edge.data.uniform_(-stdv_edge, stdv_edge)
        elif type_init == 'gaussian':
            self.mask_edge.data.normal_(0., .1)
            self.mask_node.data.normal_(0., .1)


    def original_prediction(self):
        pred = F.softmax(self.model(self.graph, self.features_ori), dim=1)[self.idx_node]
        label = np.argmax(pred.cpu().detach().numpy())
        pr_ori = pred[label]
        # print("index: {}, label: {}, pred: {}".format(self.idx_node, label, pred))
        return label, pr_ori

    def forward(self, features_cf=None, n_0=2, n_1=2, n_2=2, epsilon=0.025):
        if self.type_ex == 'e':
            pr_edge_suff, pr_edge_nec = 0, 0
            mask_edge_sigmoid = torch.sigmoid(self.mask_edge)
            if self.objective == 'ps' or self.objective == 'pns':
                pred_edge_suff = get_logit_with_mask(model=self.model,
                                                     graph=self.graph,
                                                     features=self.features_ori,
                                                     mask=mask_edge_sigmoid,
                                                     apply_softmax=True)[self.idx_node]
                # pred_edge_suff = F.softmax(self.model(self.graph, self.features_ori, eweight=mask_edge_sigmoid), dim=1)[self.idx_node]
                pr_edge_suff = pred_edge_suff[self.label]
                # print('pr reqgrad:', pr_edge_suff.requires_grad)

            if self.objective == 'pn' or self.objective == 'pns':
                for i in range(n_1):
                    noise_e = (torch.rand(mask_edge_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_e = torch.normal(0, 0.001, mask_edge_sigmoid.shape).to(self.device)
                    pred_edge_nec = get_logit_with_mask(model=self.model,
                                                        graph=self.graph,
                                                        features=self.features_ori,
                                                        mask=(1-mask_edge_sigmoid+noise_e),
                                                        apply_softmax=True)[self.idx_node]
                    # pred_edge_nec = F.softmax(self.model(self.graph, self.features_ori, eweight=(1-mask_edge_sigmoid+noise_e)), dim=1)[self.idx_node]
                    pr_edge_nec = pr_edge_nec + pred_edge_nec[self.label]
                pr_edge_nec = pr_edge_nec / n_1
                # print('pr reqgrad:', pr_edge_nec.requires_grad)
            return pr_edge_suff, pr_edge_nec, mask_edge_sigmoid

        elif self.type_ex == 'f':
            pr_node_suff, pr_node_nec = 0, 0

            identity = torch.eye(self.num_nodes).to(self.device)
            mask_node_sigmoid = torch.sigmoid(self.mask_node)

            if self.objective == 'ps' or self.objective == 'pns':
                feat_suff = mask_node_sigmoid * identity @ self.features_ori + (
                            1 - mask_node_sigmoid) * identity @ features_cf
                pred_node_suff = get_logit_with_mask(model=self.model,
                                                     graph=self.graph,
                                                     features=feat_suff,
                                                     mask=None,
                                                     apply_softmax=True)[self.idx_node]
                # pred_node_suff = F.softmax(self.model(self.graph, feat_suff), dim=1)[self.idx_node]
                pr_node_suff = pred_node_suff[self.label]

            if self.objective == 'pn' or self.objective == 'pns':
                for i in range(n_2):
                    noise_f = (torch.rand(mask_node_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_f = torch.normal(0, 0.001, mask_node_sigmoid.shape).to(self.device)
                    feat_nec = (mask_node_sigmoid - noise_f) * identity @ features_cf + (
                                1 - mask_node_sigmoid + noise_f) * identity @ self.features_ori
                    pred_node_nec = get_logit_with_mask(model=self.model,
                                                        graph=self.graph,
                                                        features=feat_nec,
                                                        mask=None,
                                                        apply_softmax=True)[self.idx_node]
                    # pred_node_nec = F.softmax(self.model(self.graph, feat_nec), dim=1)[self.idx_node]
                    pr_node_nec += pred_node_nec[self.label]
                pr_node_nec = pr_node_nec / n_2
            return pr_node_suff, pr_node_nec, mask_node_sigmoid

        elif self.type_ex == 'ef':
            pr_suff, pr_nec = 0, 0
            mask_edge_sigmoid = torch.sigmoid(self.mask_edge)
            identity = torch.eye(self.num_nodes).to(self.device)
            mask_node_sigmoid = torch.sigmoid(self.mask_node)
            if self.objective == 'ps' or self.objective == 'pns':

                feat_suff = mask_node_sigmoid * identity @ self.features_ori + (
                        1 - mask_node_sigmoid) * identity @ features_cf
                pred_suff = get_logit_with_mask(model=self.model,
                                                graph=self.graph,
                                                features=feat_suff,
                                                mask=mask_edge_sigmoid,
                                                apply_softmax=True)[self.idx_node]
                # pred_suff = F.softmax(self.model(self.graph, feat_suff, eweight=mask_edge_sigmoid),
                #                            dim=1)[self.idx_node]
                pr_suff = pred_suff[self.label]

            if self.objective == 'pn' or self.objective == 'pns':
                pr_nec_00 = 0
                pr_nec_01 = 0
                pr_nec_10 = 0
                feat_suff = mask_node_sigmoid * identity @ self.features_ori + (
                        1 - mask_node_sigmoid) * identity @ features_cf
                for i in range(n_0):
                    noise_e = (torch.rand(mask_edge_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    noise_f = (torch.rand(mask_node_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    feat_nec = (mask_node_sigmoid - noise_f) * identity @ features_cf + (
                                1 - mask_node_sigmoid + noise_f) * identity @ self.features_ori
                    pred_nec_00 = get_logit_with_mask(model=self.model,
                                                      graph=self.graph,
                                                      features=feat_nec,
                                                      mask=(1 - mask_edge_sigmoid + noise_e),
                                                      apply_softmax=True)[self.idx_node]
                    # pred_nec_00 = F.softmax(self.model(self.graph, feat_nec, eweight=(1 - mask_edge_sigmoid + noise_e)), dim=1)[self.idx_node]
                    pr_nec_00 += pred_nec_00[self.label]

                for j in range(n_1):
                    noise_e = (torch.rand(mask_edge_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    pred_nec_01 = get_logit_with_mask(model=self.model,
                                                      graph=self.graph,
                                                      features=feat_suff,
                                                      mask=(1 - mask_edge_sigmoid + noise_e),
                                                      apply_softmax=True)[self.idx_node]
                    # pred_nec_01 = F.softmax(self.model(self.graph, feat_suff, eweight=(1 - mask_edge_sigmoid + noise_e)), dim=1)[self.idx_node]
                    pr_nec_01 += pred_nec_01[self.label]

                for k in range(n_2):
                    noise_f = (torch.rand(mask_node_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_f = torch.normal(0, 0.005, mask_node_sigmoid.shape).to(self.device)
                    feat_nec = (mask_node_sigmoid - noise_f) * identity @ features_cf + (
                            1 - mask_node_sigmoid + noise_f) * identity @ self.features_ori
                    pred_nec_10 = get_logit_with_mask(model=self.model,
                                                      graph=self.graph,
                                                      features=feat_nec,
                                                      mask=mask_edge_sigmoid,
                                                      apply_softmax=True)[self.idx_node]
                    # pred_nec_10 = F.softmax(self.model(self.graph, feat_nec, eweight=mask_edge_sigmoid), dim=1)[self.idx_node]
                    pr_nec_10 += pred_nec_10[self.label]

                pr_nec = (pr_nec_00 + pr_nec_01 + pr_nec_10) / (n_0 + n_1 + n_2)

            return pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid


    def loss_e_or_f(self, pr_suff, pr_nec, mask_sigmoid):
        if self.objective == 'pns':
            pred_loss = -(pr_suff - pr_nec)
        elif self.objective == 'pn':
            pred_loss = -(1 - pr_nec)
        elif self.objective == 'ps':
            pred_loss = -(pr_suff)

        l1_loss = torch.sum(mask_sigmoid)

        eps = 1e-15
        ent = - mask_sigmoid * torch.log(mask_sigmoid + eps) - \
              (1 - mask_sigmoid) * torch.log(1 - mask_sigmoid + eps)
        if self.type_ex == 'e':
            loss =  pred_loss + self.alpha_e * l1_loss + self.beta_e * ent.mean()
        elif self.type_ex == 'f':
            loss = pred_loss + self.alpha_f * l1_loss + self.beta_f * ent.mean()

        return loss


    def loss_ef(self, pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid):
        if self.objective == 'pns':
            pred_loss = -(pr_suff - pr_nec)
        elif self.objective == 'pn':
            pred_loss = -(1 - pr_nec)
        elif self.objective == 'ps':
            pred_loss = -pr_suff

        l1_edge_loss = torch.sum(mask_edge_sigmoid)
        eps = 1e-15
        ent_edge = - mask_edge_sigmoid * torch.log(mask_edge_sigmoid + eps) - (1 - mask_edge_sigmoid) * torch.log(1 - mask_edge_sigmoid + eps)

        l1_node_loss = torch.sum(mask_node_sigmoid)
        ent_node = - mask_node_sigmoid * torch.log(mask_node_sigmoid + eps) - (1 - mask_node_sigmoid) * torch.log(1 - mask_node_sigmoid + eps)

        loss = pred_loss + self.alpha_e * l1_edge_loss + self.beta_e * ent_edge.mean() + self.alpha_f * l1_node_loss + self.beta_f * ent_node.mean()

        return loss


class ExplainerModuleGraph(nn.Module):
    def __init__(self, graph, model, features_ori, device, alpha_e, beta_e, alpha_f, beta_f, objective='pns', type_ex='e'):
        """

        :param graph: dgl graph
        :param model: fixed GNN model, dgl
        :param features_ori: original features tensor
        """
        super(ExplainerModuleGraph, self).__init__()
        self.type_ex = type_ex
        self.objective = objective
        self.graph = graph.to(device)
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.features_ori = features_ori.to(device)

        self.num_edges = graph.number_of_edges()
        self.mask_edge = nn.Parameter(torch.FloatTensor(self.num_edges, ).to(device))
        self.num_nodes = graph.number_of_nodes()
        self.mask_node = nn.Parameter(torch.FloatTensor(self.num_nodes, ).to(device))

        self.reset_parameter()
        self.label, self.pr_ori = self.original_prediction()
        self.device = device

        self.alpha_e = alpha_e
        self.beta_e = beta_e
        self.alpha_f = alpha_f
        self.beta_f = beta_f

        # print('label: {}. pr_ori: {}'.format(self.pr_ori))

    def reset_parameter(self, type_init='uniform'):
        if type_init == 'uniform':
            stdv_node = 1. / math.sqrt(self.mask_node.shape[0])
            stdv_edge = 1. / math.sqrt(self.mask_edge.shape[0])
            self.mask_node.data.uniform_(-stdv_node, stdv_node)
            self.mask_edge.data.uniform_(-stdv_edge, stdv_edge)
        elif type_init == 'gaussian':
            self.mask_edge.data.normal_(0.0, .1)
            self.mask_node.data.normal_(0.0, .1)


    def original_prediction(self):
        pred = F.softmax(self.model(self.graph, self.features_ori), dim=1)
        label = np.argmax(pred.cpu().detach().numpy())
        pr_ori = pred[0, label]
        # print("label: {}, pred: {}".format(label, pred))
        return label, pr_ori

    def forward(self, features_cf=None, n_0=2, n_1=2, n_2=2, epsilon=0.025):
        if self.type_ex == 'e':
            pr_edge_suff, pr_edge_nec = 0, 0
            mask_edge_sigmoid = torch.sigmoid(self.mask_edge)

            if self.objective == 'ps' or self.objective == 'pns':
                pred_edge_suff = get_logit_with_mask(model=self.model,
                                                     graph=self.graph,
                                                     features=self.features_ori,
                                                     mask=mask_edge_sigmoid,
                                                     apply_softmax=True)
                # pred_edge_suff = F.softmax(self.model(self.graph, self.features_ori, eweight=mask_edge_sigmoid), dim=1)
                pr_edge_suff = pred_edge_suff[0, self.label]

            if self.objective == 'pn' or self.objective == 'pns':
                pr_edge_nec = 0
                for i in range(n_1):
                    noise_e = (torch.rand(mask_edge_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    pred_edge_nec = get_logit_with_mask(model=self.model,
                                                        graph=self.graph,
                                                        features=self.features_ori,
                                                        mask=(1-mask_edge_sigmoid+noise_e),
                                                        apply_softmax=True)
                    # pred_edge_nec = F.softmax(self.model(self.graph, self.features_ori, eweight=(1-mask_edge_sigmoid+noise_e)), dim=1)
                    pr_edge_nec += pred_edge_nec[0, self.label]

                pr_edge_nec = pr_edge_nec / n_1

            return pr_edge_suff, pr_edge_nec, mask_edge_sigmoid

        elif self.type_ex == 'f':
            pr_node_suff, pr_node_nec = 0, 0

            identity = torch.eye(self.num_nodes).to(self.device)
            mask_node_sigmoid = torch.sigmoid(self.mask_node)

            if self.objective == 'ps' or self.objective == 'pns':

                feat_suff = mask_node_sigmoid * identity @ self.features_ori + (1 - mask_node_sigmoid) * identity @ features_cf
                pred_node_suff = get_logit_with_mask(model=self.model,
                                                     graph=self.graph,
                                                     features=feat_suff,
                                                     mask=None,
                                                     apply_softmax=True)
                # pred_node_suff = F.softmax(self.model(self.graph, feat_suff), dim=1)
                pr_node_suff = pred_node_suff[0, self.label]

            if self.objective == 'pn' or self.objective == 'pns':
                pr_node_nec = 0
                for i in range(n_2):
                    noise_f = (torch.rand(mask_node_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_f = torch.normal(0, 0.001, mask_node_sigmoid.shape).to(self.device)

                    feat_nec = (mask_node_sigmoid-noise_f) * identity @ features_cf + (1 - mask_node_sigmoid + noise_f) * identity @ self.features_ori
                    pred_node_nec = get_logit_with_mask(model=self.model,
                                                        graph=self.graph,
                                                        features=feat_nec,
                                                        mask=None,
                                                        apply_softmax=True)
                    # pred_node_nec = F.softmax(self.model(self.graph, feat_nec), dim=1)
                    pr_node_nec += pred_node_nec[0, self.label]

                pr_node_nec = pr_node_nec / n_2

            return pr_node_suff, pr_node_nec, mask_node_sigmoid

        elif self.type_ex == 'ef':
            pr_suff, pr_nec = 0, 0
            mask_edge_sigmoid = torch.sigmoid(self.mask_edge)
            identity = torch.eye(self.num_nodes).to(self.device)
            mask_node_sigmoid = torch.sigmoid(self.mask_node)
            if self.objective == 'ps' or self.objective == 'pns':

                feat_suff = mask_node_sigmoid * identity @ self.features_ori + (
                        1 - mask_node_sigmoid) * identity @ features_cf
                pred_suff = get_logit_with_mask(model=self.model,
                                                graph=self.graph,
                                                features=feat_suff,
                                                mask=mask_edge_sigmoid,
                                                apply_softmax=True)
                # pred_suff = F.softmax(self.model(self.graph, feat_suff, eweight=mask_edge_sigmoid),
                #                            dim=1)
                pr_suff = pred_suff[0, self.label]

            if self.objective == 'pn' or self.objective == 'pns':
                pr_nec_00 = 0
                pr_nec_01 = 0
                pr_nec_10 = 0
                feat_suff = mask_node_sigmoid * identity @ self.features_ori + (
                        1 - mask_node_sigmoid) * identity @ features_cf
                for i in range(n_0):
                    noise_e = (torch.rand(mask_edge_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    noise_f = (torch.rand(mask_node_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_e = torch.normal(0, 0.001, mask_edge_sigmoid.shape).to(self.device)
                    # noise_f = torch.normal(0, 0.001, mask_node_sigmoid.shape).to(self.device)
                    feat_nec = (mask_node_sigmoid - noise_f) * identity @ features_cf + (
                                1 - mask_node_sigmoid + noise_f) * identity @ self.features_ori
                    pred_nec_00 = get_logit_with_mask(model=self.model,
                                                      graph=self.graph,
                                                      features=feat_nec,
                                                      mask=(1 - mask_edge_sigmoid + noise_e),
                                                      apply_softmax=True)
                    # pred_nec_00 = F.softmax(self.model(self.graph, feat_nec, eweight=(1 - mask_edge_sigmoid + noise_e)), dim=1)
                    pr_nec_00 += pred_nec_00[0, self.label]

                for j in range(n_1):
                    noise_e = (torch.rand(mask_edge_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_e = torch.normal(0, 0.001, mask_edge_sigmoid.shape).to(self.device)
                    pred_nec_01 = get_logit_with_mask(model=self.model,
                                                      graph=self.graph,
                                                      features=feat_suff,
                                                      mask=(1 - mask_edge_sigmoid + noise_e),
                                                      apply_softmax=True)
                    # pred_nec_01 = F.softmax(self.model(self.graph, feat_suff, eweight=(1 - mask_edge_sigmoid + noise_e)), dim=1)
                    pr_nec_01 += pred_nec_01[0, self.label]

                for k in range(n_2):
                    noise_f = (torch.rand(mask_node_sigmoid.shape).to(self.device) - 0.5) * 2 * epsilon
                    # noise_f = torch.normal(0, 0.001, mask_node_sigmoid.shape).to(self.device)
                    feat_nec = (mask_node_sigmoid - noise_f) * identity @ features_cf + (1 - mask_node_sigmoid + noise_f) * identity @ self.features_ori
                    pred_nec_10 = get_logit_with_mask(model=self.model,
                                                      graph=self.graph,
                                                      features=feat_nec,
                                                      mask=mask_edge_sigmoid,
                                                      apply_softmax=True)
                    # pred_nec_10 = F.softmax(self.model(self.graph, feat_nec, eweight=mask_edge_sigmoid), dim=1)
                    pr_nec_10 += pred_nec_10[0, self.label]

                pr_nec = (pr_nec_00 + pr_nec_01 + pr_nec_10) / (n_0 + n_1 + n_2)

            return pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid

    def loss_e_or_f(self, pr_suff, pr_nec, mask_sigmoid):
        if self.objective == 'pns':
            pred_loss = -(pr_suff - pr_nec)
        elif self.objective == 'pn':
            pred_loss = -(1 - pr_nec)
        elif self.objective == 'ps':
            pred_loss = -pr_suff

        l1_edge_loss = torch.sum(mask_sigmoid)
        eps = 1e-15
        ent = - mask_sigmoid * torch.log(mask_sigmoid + eps) - (1 - mask_sigmoid) * torch.log(1 - mask_sigmoid + eps)
        if self.type_ex == 'e':
            loss = pred_loss + self.alpha_e * l1_edge_loss + self.beta_e * ent.mean()
        elif self.type_ex == 'f':
            loss = pred_loss + self.alpha_f * l1_edge_loss + self.beta_f * ent.mean()

        return loss


    def loss_ef(self, pr_suff, pr_nec, mask_edge_sigmoid, mask_node_sigmoid):
        if self.objective == 'pns':
            pred_loss = -(pr_suff - pr_nec)
        elif self.objective == 'pn':
            pred_loss = -(1 - pr_nec)
        elif self.objective == 'ps':
            pred_loss = -pr_suff

        l1_edge_loss = torch.sum(mask_edge_sigmoid)
        eps = 1e-15
        ent_edge = - mask_edge_sigmoid * torch.log(mask_edge_sigmoid + eps) - (1 - mask_edge_sigmoid) * torch.log(1 - mask_edge_sigmoid + eps)

        l1_node_loss = torch.sum(mask_node_sigmoid)
        ent_node = - mask_node_sigmoid * torch.log(mask_node_sigmoid + eps) - (1 - mask_node_sigmoid) * torch.log(1 - mask_node_sigmoid + eps)

        loss = pred_loss + self.alpha_e * l1_edge_loss + self.beta_e * ent_edge.mean() + self.alpha_f * l1_node_loss + self.beta_f * ent_node.mean()

        return loss


def get_logit_with_mask(model, graph, features, mask=None, apply_softmax=False):
    model.eval()
    logit = model(graph, features, eweight=mask)
    if apply_softmax:
        logit = F.softmax(logit, dim=1)
    return logit