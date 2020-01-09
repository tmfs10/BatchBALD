import torch
import numpy as np
import torch.nn as nn

from blackhc.progress_bar import with_progress_bar

import torch.distributions as tdist
import joint_entropy.exact as joint_entropy_exact
import joint_entropy.sampling as joint_entropy_sampling
import torch_utils
import math
import time
import hsic

from acquisition_batch import AcquisitionBatch

from acquisition_functions import AcquisitionFunction
from reduced_consistent_mc_sampler import reduced_eval_consistent_bayesian_model


compute_multi_bald_bag_multi_bald_batch_size = None

def compute_multi_detk_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    hsic_compute_batch_size,
    device=None,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    sample_B_K_C = result.logits_B_K_C 

    """
    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample
    """

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    winner_index = result.scores_B.argmax().item()

    ack_bag = [winner_index]
    acquisition_bag_scores = [0]
    batch_B_K_C = samples_B_K_C[winner_index:winner_index+1] # cur_ack_size, K, C
    print('Computing HSIC for', B, 'points')
    for ackb_i in range(1, b):
        detk_scores = []
        bs = 0
        bi = 0
        #print('batch_kernel.shape', batch_kernel.shape)
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs
            #print(batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]).shape)
            #print(kernel_matrices[bs:be].unsqueeze(-1).shape)
            detk_scores += [ops.parallel_cross_cov(
                batch_kernel.permute([2, 1, 0]),
                samples_B_K_C[bs:be].permute([2, 1, 0]),
                ).mean(dim=0).max(dim=1)[0]]

            bs = be
        detk_scores = torch.cat(detk_scores, dim=0).mean(dim=0)
        hsic_scores[ack_bag] = -math.inf
        
        winner_index = hsic_scores.argmax().item() # choose the most independent set of items
        assert winner_index not in ack_bag
        winner_hsic = hsic_scores[winner_index].item()

        print('hsic of batch', winner_hsic, 'for batch size', ackb_i+1)

        #print('adding', kernel_matrices[winner_index].unsqueeze(-1).shape)
        batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)], dim=-1)
        ack_bag += [winner_index]
        acquisition_bag_scores += [winner_hsic]

    np.set_printoptions(precision=3, suppress=True)
    print('Acquired predictions')
    for i in range(len(ack_bag)):
        print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    return AcquisitionBatch(ack_bag, acquisition_bag_scores, None)

def compute_multi_hsic_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    hsic_compute_batch_size,
    hsic_kernel_name,
    device=None,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    sample_B_K_C = probs_B_K_C
    #sample_B_K_C = result.logits_B_K_C

    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    winner_index = result.scores_B.argmax().item()

    ack_bag = [winner_index]
    acquisition_bag_scores = [0]
    batch_kernel = kernel_matrices[winner_index:winner_index+1].permute([1, 2, 0]) # n, n, cur_ack_size
    print('Computing HSIC for', B, 'points')
    batch_hsic_scores = [0.]
    for ackb_i in range(1, b):
        hsic_scores = [[] for i in range(len(ack_bag))]
        bs = 0
        bi = 0
        #print('batch_kernel.shape', batch_kernel.shape)
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs
            #print(batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]).shape)
            #print(kernel_matrices[bs:be].unsqueeze(-1).shape)
            """
            for bi2 in range(bs, be):
                hsic_scores += [hsic.total_hsic(
                    torch.cat(
                        [
                            batch_kernel,
                            kernel_matrices[bi2].unsqueeze(-1),
                        ],
                        dim=-1)
                    )]
            hsic_scores += [hsic.total_hsic_parallel(
                torch.cat(
                    [
                        batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]), 
                        kernel_matrices[bs:be].unsqueeze(-1)
                    ],
                    dim=-1)
                #.mean(dim=-1).unsqueeze(-1).repeat([1, 1, 1, 2])
                )]
            """

            for ackb_i2 in range(len(ack_bag)):
                ack_kernel = batch_kernel[:, :, ackb_i2:ackb_i2+1]
                hsic_scores[ackb_i2] += [hsic.total_hsic_parallel(
                    torch.cat(
                        [
                            ack_kernel.unsqueeze(0).repeat([m, 1, 1, 1]), 
                            kernel_matrices[bs:be].unsqueeze(-1)
                        ],
                        dim=-1)
                    .mean(dim=-1).unsqueeze(-1).repeat([1, 1, 1, 2])
                    )]

            bs = be
        for ackb_i2 in range(len(ack_bag)):
            hsic_scores[ackb_i2] = torch.cat(hsic_scores[ackb_i2]).unsqueeze(0)
        hsic_scores = torch.cat(hsic_scores, dim=0)
        #hsic_scores[:, ack_bag] = -math.inf
        hsic_scores = hsic_scores.min(dim=0)[0]
        hsic_scores[ack_bag] = -math.inf
        
        winner_index = hsic_scores.argmax().item() # choose the most independent set of items
        assert winner_index not in ack_bag
        winner_hsic = hsic_scores[winner_index].item()
        batch_hsic_scores += [winner_hsic]

        print('hsic of batch', winner_hsic, 'with score', result.scores_B[winner_index].item(), 'for batch size', ackb_i+1)

        #print('adding', kernel_matrices[winner_index].unsqueeze(-1).shape)
        batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)], dim=-1)
        ack_bag += [winner_index]
        acquisition_bag_scores += [winner_hsic]

    np.set_printoptions(precision=3, suppress=True)
    print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    return AcquisitionBatch(ack_bag, acquisition_bag_scores, None)


class ProjectedFrankWolfe(object):
    def __init__(self, py, logits, J, **kwargs):
        """
        Constructs a batch of points using ACS-FW with random projections. Note the slightly different interface.
        :param data: (ActiveLearningDataset) Dataset.
        :param J: (int) Number of projections.
        :param kwargs: (dict) Additional arguments.
        """
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.device = logits.device

        self.ELn, self.entropy = self.get_projections(py, logits, J, **kwargs)
        squared_norm = torch.sum(self.ELn * self.ELn, dim=-1)
        self.sigmas = torch.sqrt(squared_norm + 1e-6)
        self.sigma = self.sigmas.sum()
        self.EL = torch.sum(self.ELn, dim=0)

    def get_projections(self, py, logits, J, projection='two', gamma=0, **kwargs):
        """
        Get projections for ACS approximate procedure
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        assert logits.shape[1] == J
        C = logits.shape[-1]
        ent = lambda py: torch.distributions.Categorical(probs=py).entropy()
        projections = []
        feat_x = []
        with torch.no_grad():
            ent_x = ent(py)
            if projection == 'two':
                for j in range(J):
                    cur_logits = logits[:, j, :]
                    ys = torch.ones_like(cur_logits).type(torch.LongTensor) * torch.arange(C, device=logits.device)[None, :]
                    ys = ys.t()
                    loglik = torch.stack([-self.cross_entropy(cur_logits, y) for y in ys]).t()
                    loglik = torch.sum(py * loglik, dim=-1, keepdim=True)
                    projections.append(loglik + gamma * ent_x[:, None])
            else:
                raise NotImplementedError

        return torch.sqrt(1 / torch.DoubleTensor([J], device=logits.device)) * torch.cat(projections, dim=1), ent_x

    def _init_build(self, M, **kwargs):
        pass  # unused

    def build(self, M=1, **kwargs):
        """
        Constructs a batch of points to sample from the unlabeled set.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional parameters.
        :return: (list of ints) Selected data point indices.
        """
        self._init_build(M, **kwargs)
        w = torch.zeros([len(self.ELn), 1], device=self.device).double()
        norm = lambda weights: (self.EL - (self.ELn.t() @ weights).squeeze()).norm()
        for m in range(M):
            w = self._step(m, w)

        # print(w[w.nonzero()[:, 0]].cpu().numpy())
        print('|| L-L(w)  ||: {:.4f}'.format(norm(w)))
        print('|| L-L(w1) ||: {:.4f}'.format(norm((w > 0).double())))
        print('Avg pred entropy (pool): {:.4f}'.format(self.entropy.mean().item()))
        print('Avg pred entropy (batch): {:.4f}'.format(self.entropy[w.flatten() > 0].mean().item()))

        return w.nonzero()[:, 0].cpu().numpy()

    def _step(self, m, w, **kwargs):
        """
        Applies one step of the Frank-Wolfe algorithm to update weight vector w.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        #print(self.ELn.type(), w.type())
        self.ELw = (self.ELn.t() @ w).squeeze()
        scores = (self.ELn / self.sigmas[:, None]) @ (self.EL - self.ELw)
        f = torch.argmax(scores)
        gamma, f1 = self.compute_gamma(f, w)
        # print('f: {}, gamma: {:.4f}, score: {:.4f}'.format(f, gamma.item(), scores[f].item()))
        if np.isnan(gamma.cpu()):
            raise ValueError

        w = (1 - gamma) * w + gamma * (self.sigma / self.sigmas[f]) * f1
        return w

    def compute_gamma(self, f, w):
        """
        Computes line-search parameter gamma.
        :param f: (int) Index of selected data point.
        :param w: (numpy array) Current weight vector.
        :return: (float, numpy array) Line-search parameter gamma and f-th unit vector [0, 0, ..., 1, ..., 0]
        """
        f1 = torch.zeros_like(w)
        f1[f] = 1
        Lf = (self.sigma / self.sigmas[f] * f1.t() @ self.ELn).squeeze()
        Lfw = Lf - self.ELw
        numerator = Lfw @ (self.EL - self.ELw)
        denominator = Lfw @ Lfw
        return numerator / denominator, f1

def compute_acs_fw_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_entropy_bag_size,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    B, K, C = list(result.logits_B_K_C.shape) # (pool size, mc dropout samples, classes)
    py = result.logits_B_K_C.exp_().mean(dim=1)

    num_projections=10
    gamma=0.7
    assert K >= num_projections
    cs = ProjectedFrankWolfe(py, result.logits_B_K_C[:, :num_projections, :], num_projections, gamma=gamma)

    end_time = time.process_time()
    print('ack time taken', time_taken)

    global_acquisition_bag = cs.build(b)
    time_taken = end_time-start_time
    acquisition_bag_scores = []

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_fass_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_entropy_bag_size,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    B, K, C = list(result.logits_B_K_C.shape)
    probs_B_C = result.logits_B_K_C.exp_().mean(dim=1)
    preds_B = probs_B_C.max(dim=-1)[1]
    entropy = -(probs_B_C * probs_B_C.log()).sum(dim=-1)

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []

    score_sort = torch.sort(entropy, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]

    cand_pts_idx = set(score_sort_idx[:max_entropy_bag_size].cpu().numpy().tolist())

    cand_X = []
    cand_X_preds = []
    cand_X_idx = []
    for i, (batch, labels) in enumerate(
        with_progress_bar(available_loader, unit_scale=available_loader.batch_size)
    ):
        lower = i * available_loader.batch_size
        upper = min(lower + available_loader.batch_size, B)
        idx_to_extract = np.array(list(set(range(lower, upper)).intersection(cand_pts_idx)), dtype=np.int32)
        cand_X_preds += [preds_B[idx_to_extract]]
        cand_X_idx += [torch.from_numpy(idx_to_extract).long()]
        idx_to_extract -= lower

        batch = batch.view(batch.shape[0], -1) # batch_size x num_features
        cand_X += [batch[idx_to_extract]]

    cand_X = torch.cat(cand_X, dim=0).unsqueeze(1).to(device)
    cand_X_preds = torch.cat(cand_X_preds, dim=0).to(device)
    cand_X_idx = torch.cat(cand_X_idx, dim=0).to(device)
    sqdist = hsic.sqdist(cand_X, cand_X).mean(-1) # cand_X size x cand_X size
    max_dist = sqdist.max()

    cand_min_dist = torch.ones((cand_X.shape[0],), device=device) * max_dist
    ack_bag = []
    global_acquisition_bag = []
    for ackb_i in range(b):
        cand_distance = torch.ones((cand_X.shape[0],), device=device) * max_dist
        for c in range(C):
            cand_c_idx = cand_X_preds == c
            if cand_c_idx.long().sum() == 0:
                continue
            cand_distance[cand_c_idx] = torch.min(torch.cat([cand_min_dist[cand_c_idx].unsqueeze(-1).repeat([1, sqdist.shape[1]]).unsqueeze(-1), sqdist[cand_c_idx].unsqueeze(-1)], dim=-1), dim=-1)[0].mean(1)
        cand_distance[ack_bag] = max_dist
        winner_index = cand_distance.argmin().item()
        ack_bag += [winner_index]
        #print('cand_distance.shape', cand_distance.shape, winner_index, cand_X_idx.shape)
        winner_index = cand_X_idx[winner_index].item()
        global_acquisition_bag.append(result.subset_split.get_dataset_indices([winner_index]).item())

    assert len(ack_bag) == b
    np.set_printoptions(precision=3, suppress=True)
    #print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_ical_hsic_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    hsic_compute_batch_size,
    hsic_kernel_name,
    hsic_resample=True,
    device=None,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    #sample_B_K_C = probs_B_K_C
    sample_B_K_C = result.logits_B_K_C

    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample#.to(device)

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)#.to(device)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]).to(device) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []
    batch_kernel = None
    print('Computing HSIC for', B, 'points')

    num_to_condense = 200

    score_sort = torch.sort(result.scores_B, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]
    #indices_to_condense = [idx.item() for idx in score_sort_idx[:num_to_condense]]
    indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    for ackb_i in range(b):
        bs = 0
        hsic_scores = []
        condense_kernels = kernel_matrices[indices_to_condense].permute([1, 2, 0]).mean(dim=-1, keepdim=True).unsqueeze(0) # 1, K, K, 1
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs

            if batch_kernel is None:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]),
                        kernel_matrices[bs:be].unsqueeze(-1),
                    ],
                    dim=-1
                    ).to(device)
                )]
            else:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]),
                        torch.cat([
                            batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]),
                            kernel_matrices[bs:be].unsqueeze(-1),
                        ], dim=-1).mean(dim=-1, keepdim=True),
                    ],
                    dim=-1
                    ).to(device)
                )]
            bs = be

        hsic_scores = torch.cat(hsic_scores)
        hsic_scores[ack_bag] = -math.inf

        winner_index = hsic_scores.argmax().item()
        ack_bag += [winner_index]
        global_acquisition_bag.append(result.subset_split.get_dataset_indices([winner_index]).item())
        acquisition_bag_scores += [hsic_scores[winner_index].item()]
        print('winner score', result.scores_B[winner_index].item(), ', hsic_score', hsic_scores[winner_index].item(), ', ackb_i', ackb_i)
        if batch_kernel is None:
            batch_kernel = kernel_matrices[winner_index].unsqueeze(-1)
        else:
            batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)], dim=-1)

        result.scores_B[winner_index] = -math.inf
        score_sort = torch.sort(result.scores_B, descending=True)
        score_sort_idx = score_sort[1]
        score_sort = score_sort[0]
        #indices_to_condense = [idx.item() for idx in score_sort_idx[:num_to_condense]]
        if hsic_resample:
            indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    assert len(ack_bag) == b
    np.set_printoptions(precision=3, suppress=True)
    #print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_ical_hsic_batch_scale(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_batch_compute_size,
    hsic_compute_batch_size,
    hsic_kernel_name,
    hsic_resample=True,
    device=None,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    #sample_B_K_C = probs_B_K_C
    sample_B_K_C = result.logits_B_K_C

    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample#.to(device)

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)#.to(device)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]).to(device) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []
    batch_kernel = None
    print('Computing HSIC for', B, 'points')

    num_to_condense = 200

    score_sort = torch.sort(result.scores_B, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]
    #indices_to_condense = [idx.item() for idx in score_sort_idx[:num_to_condense]]
    indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    for ackb_i in range(b):
        bs = 0
        hsic_scores = []
        condense_kernels = kernel_matrices[indices_to_condense].permute([1, 2, 0]).mean(dim=-1, keepdim=True).unsqueeze(0) # 1, K, K, 1
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs

            if batch_kernel is None:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]),
                        kernel_matrices[bs:be].unsqueeze(-1),
                    ],
                    dim=-1
                    ).to(device)
                )]
            else:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]),
                        torch.cat([
                            batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]),
                            kernel_matrices[bs:be].unsqueeze(-1),
                        ], dim=-1).mean(dim=-1, keepdim=True),
                    ],
                    dim=-1
                    ).to(device)
                )]
            bs = be

        hsic_scores = torch.cat(hsic_scores)
        hsic_scores[ack_bag] = -math.inf

        winner_index = hsic_scores.argmax().item()
        ack_bag += [winner_index]
        global_acquisition_bag.append(result.subset_split.get_dataset_indices([winner_index]).item())
        acquisition_bag_scores += [hsic_scores[winner_index].item()]
        print('winner score', result.scores_B[winner_index].item(), ', hsic_score', hsic_scores[winner_index].item(), ', ackb_i', ackb_i)
        if batch_kernel is None:
            batch_kernel = kernel_matrices[winner_index].unsqueeze(-1) # K, K, 1
        else:
            batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)], dim=-1) # K, K, ack_size
            assert len(batch_kernel.shape) == 3
            if batch_kernel.shape[-1] >= max_batch_compute_size and max_batch_compute_size != None:
                idxes = np.random.choice(batch_kernel.shape[-1], size=max_batch_compute_size, replace=False)
                batch_kernel = batch_kernel[:, :, idxes]

        result.scores_B[winner_index] = -math.inf
        score_sort = torch.sort(result.scores_B, descending=True)
        score_sort_idx = score_sort[1]
        score_sort = score_sort[0]
        #indices_to_condense = [idx.item() for idx in score_sort_idx[:num_to_condense]]
        if hsic_resample:
            indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    assert len(ack_bag) == b
    np.set_printoptions(precision=3, suppress=True)
    #print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_multi_hsic_batch2(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    hsic_compute_batch_size,
    hsic_kernel_name,
    device=None,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    sample_B_K_C = probs_B_K_C
    #sample_B_K_C = result.logits_B_K_C

    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    score_sort = torch.sort(result.scores_B, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]

    winner_index = result.scores_B.argmax().item()

    ack_bag = [winner_index]
    acquisition_bag_scores = [score_sort[0].item()]
    batch_kernel = kernel_matrices[winner_index:winner_index+1].permute([1, 2, 0]) # n, n, cur_ack_size
    print('Computing HSIC for', B, 'points')

    pairwise = True

    for i in range(score_sort.shape[0]):
        if len(ack_bag) == b:
            break
        cand_pt = score_sort_idx[i].item()
        if cand_pt in ack_bag:
            continue

        if pairwise:
            hsic_scores = []
            for ackb_i in range(len(ack_bag)):
                hsic_scores += [hsic.hsic_xy(
                        batch_kernel[:, :, ackb_i],
                        kernel_matrices[cand_pt],
                        normalized=True,
                        ).item()]
            #hsic_score = sum(hsic_scores)/len(hsic_scores)
            hsic_score = sum(hsic_scores)/len(hsic_scores)
        else:
            hsic_score = hsic.hsic_xy(
                    batch_kernel.mean(-1),
                    kernel_matrices[cand_pt],
                    normalized=True,
                    ).item()

        if hsic_score >= 0.01:
            continue

        winner_index = cand_pt
        ack_bag += [winner_index]
        acquisition_bag_scores += [score_sort[i].item()]
        print('winner score', score_sort[i].item(), ', hsic_score', hsic_score, ', i', i)
        batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)], dim=-1)

    np.set_printoptions(precision=3, suppress=True)
    #print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    return AcquisitionBatch(ack_bag, acquisition_bag_scores, None)

def compute_multi_hsic_batch3(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    hsic_compute_batch_size,
    hsic_kernel_name,
    device=None,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    #sample_B_K_C = probs_B_K_C
    sample_B_K_C = result.logits_B_K_C

    """
    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample
    """

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []
    batch_kernel = None
    print('Computing HSIC for', B, 'points')
    batch_hsic_scores = []
    for ackb_i in range(b):
        if len(ack_bag) > 0:
            hsic_scores = [[] for i in range(len(ack_bag))]
        else:
            hsic_scores = []
        bs = 0
        bi = 0
        #print('batch_kernel.shape', batch_kernel.shape)
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs
            #print(batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]).shape)
            #print(kernel_matrices[bs:be].unsqueeze(-1).shape)

            if len(ack_bag) > 0:
                for ackb_i2 in range(len(ack_bag)):
                    ack_kernel = batch_kernel[:, :, ackb_i2:ackb_i2+1]
                    hsic_scores[ackb_i2] += [hsic.total_hsic_parallel(
                        torch.cat(
                            [
                                ack_kernel.unsqueeze(0).repeat([m, 1, 1, 1]), 
                                kernel_matrices[bs:be].unsqueeze(-1)
                            ],
                            dim=-1)
                        #.mean(dim=-1).unsqueeze(-1).repeat([1, 1, 1, 2])
                        )]
            else:
                hsic_scores += [hsic.total_hsic_parallel(
                    kernel_matrices[bs:be].unsqueeze(-1).repeat([1, 1, 1, 2])
                    )]

            bs = be
        for ackb_i2 in range(len(ack_bag)):
            hsic_scores[ackb_i2] = torch.cat(hsic_scores[ackb_i2]).unsqueeze(0)
        hsic_scores = torch.cat(hsic_scores, dim=0)
        #hsic_scores[:, ack_bag] = -math.inf
        if len(ack_bag) > 0:
            if False:
                hsic_scores = hsic_scores.mean(dim=0)
            else:
                hsic_scores = hsic_scores.min(dim=0)[0]
            hsic_scores[ack_bag] = -math.inf
        
        winner_index = hsic_scores.argmax().item() # choose the most independent set of items
        assert winner_index not in ack_bag
        winner_hsic = hsic_scores[winner_index].item()
        batch_hsic_scores += [winner_hsic]

        print('hsic of batch', winner_hsic, 'for batch size', ackb_i+1)

        #print('adding', kernel_matrices[winner_index].unsqueeze(-1).shape)

        if len(ack_bag) > 0:
            batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)], dim=-1)
        else:
            batch_kernel = kernel_matrices[winner_index].unsqueeze(-1)
        ack_bag += [winner_index]
        global_acquisition_bag.append(result.subset_split.get_dataset_indices([winner_index]).item())
        acquisition_bag_scores += [winner_hsic]

    np.set_printoptions(precision=3, suppress=True)
    print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None)

def compute_multi_bald_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )
    start_time = time.process_time()

    subset_split = result.subset_split

    partial_multi_bald_B = result.scores_B
    # Now we can compute the conditional entropy
    conditional_entropies_B = joint_entropy_exact.batch_conditional_entropy_B(result.logits_B_K_C)

    # We turn the logits into probabilities.
    probs_B_K_C = result.logits_B_K_C.exp_()

    # Don't need the result anymore.
    result = None

    torch_utils.gc_cuda()
    # torch_utils.cuda_meminfo()

    with torch.no_grad():
        num_samples_per_ws = 40000 // k
        num_samples = num_samples_per_ws * k

        if device.type == "cuda":
            # KC_memory = k*num_classes*8
            sample_MK_memory = num_samples * k * 8
            MC_memory = num_samples * num_classes * 8
            copy_buffer_memory = 256 * num_samples * num_classes * 8
            slack_memory = 2 * 2 ** 30
            multi_bald_batch_size = (
                torch_utils.get_cuda_available_memory() - (sample_MK_memory + copy_buffer_memory + slack_memory)
            ) // MC_memory

            global compute_multi_bald_bag_multi_bald_batch_size
            if compute_multi_bald_bag_multi_bald_batch_size != multi_bald_batch_size:
                compute_multi_bald_bag_multi_bald_batch_size = multi_bald_batch_size
                print(f"New compute_multi_bald_bag_multi_bald_batch_size = {multi_bald_batch_size}")
        else:
            multi_bald_batch_size = 16

        subset_acquisition_bag = []
        global_acquisition_bag = []
        acquisition_bag_scores = []

        # We use this for early-out in the b==0 case.
        MIN_SPREAD = 0.1

        if b == 0:
            b = 100
            early_out = True
        else:
            early_out = False

        prev_joint_probs_M_K = None
        prev_samples_M_K = None
        for i in range(b):
            torch_utils.gc_cuda()

            if i > 0:
                # Compute the joint entropy
                joint_entropies_B = torch.empty((len(probs_B_K_C),), dtype=torch.float64)

                exact_samples = num_classes ** i
                if exact_samples <= num_samples:
                    prev_joint_probs_M_K = joint_entropy_exact.joint_probs_M_K(
                        probs_B_K_C[subset_acquisition_bag[-1]][None].to(device),
                        prev_joint_probs_M_K=prev_joint_probs_M_K,
                    )

                    # torch_utils.cuda_meminfo()
                    batch_exact_joint_entropy(
                        probs_B_K_C, prev_joint_probs_M_K, multi_bald_batch_size, device, joint_entropies_B
                    )
                else:
                    if prev_joint_probs_M_K is not None:
                        prev_joint_probs_M_K = None
                        torch_utils.gc_cuda()

                    # Gather new traces for the new subset_acquisition_bag.
                    prev_samples_M_K = joint_entropy_sampling.sample_M_K(
                        probs_B_K_C[subset_acquisition_bag].to(device), S=num_samples_per_ws
                    )

                    # torch_utils.cuda_meminfo()
                    for joint_entropies_b, probs_b_K_C in with_progress_bar(
                        torch_utils.split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size),
                        unit_scale=multi_bald_batch_size,
                    ):
                        joint_entropies_b.copy_(
                            joint_entropy_sampling.batch(probs_b_K_C.to(device), prev_samples_M_K), non_blocking=True)

                        # torch_utils.cuda_meminfo()

                    prev_samples_M_K = None
                    torch_utils.gc_cuda()

                partial_multi_bald_B = joint_entropies_B - conditional_entropies_B
                joint_entropies_B = None

            # Don't allow reselection
            partial_multi_bald_B[subset_acquisition_bag] = -math.inf

            winner_index = partial_multi_bald_B.argmax().item()

            # Actual MultiBALD is:
            actual_multi_bald_B = partial_multi_bald_B[winner_index] - torch.sum(
                conditional_entropies_B[subset_acquisition_bag]
            )
            actual_multi_bald_B = actual_multi_bald_B.item()

            print(f"Actual MultiBALD: {actual_multi_bald_B}")

            # If we early out, we don't take the point that triggers the early out.
            # Only allow early-out after acquiring at least 1 sample.
            if early_out and i > 1:
                current_spread = actual_multi_bald_B[winner_index] - actual_multi_bald_B.median()
                if current_spread < MIN_SPREAD:
                    print("Early out")
                    break

            acquisition_bag_scores.append(actual_multi_bald_B)

            subset_acquisition_bag.append(winner_index)
            # We need to map the index back to the actual dataset.
            global_acquisition_bag.append(subset_split.get_dataset_indices([winner_index]).item())

            print(f"Acquisition bag: {sorted(global_acquisition_bag)}, num_ack: {i}")

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken


def batch_exact_joint_entropy(probs_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, probs_b_K_C in with_progress_bar(
        torch_utils.split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(probs_b_K_C.to(device), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b


def batch_exact_joint_entropy_logits(logits_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, logits_b_K_C in with_progress_bar(
        torch_utils.split_tensors(out_joint_entropies_B, logits_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(logits_b_K_C.to(device).exp(), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b
