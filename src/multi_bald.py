import torch
import torch.nn as nn

from blackhc.progress_bar import with_progress_bar

import torch.distributions as tdist
import joint_entropy.exact as joint_entropy_exact
import joint_entropy.sampling as joint_entropy_sampling
import torch_utils
import math
import hsic

from acquisition_batch import AcquisitionBatch

from acquisition_functions import AcquisitionFunction
from reduced_consistent_mc_sampler import reduced_eval_consistent_bayesian_model


compute_multi_bald_bag_multi_bald_batch_size = None

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
	"""
	dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
	sample_B_K_C = dist.sample([1]) # shape 1 x B*K
	assert list(sample_B_K_C.shape) == [1, B*K]
	sample_B_K_C = sample_B_K_C[0]
	oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
	oh_sample = oh_sample.view(B, K, C)
	"""

	kernel_fn = getattr(hsic, 'dimwise_'+hsic_kernel_name+'_kernels')

	dist_matrices = []
	bs = 0
	while bs < B:
		be = min(B, bs+hsic_compute_batch_size)
		dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2]))
		dist_matrices += [dist_matrix]
		bs = be
	dist_matrices = torch.cat(dist_matrices, dim=-1)

	bs = 0
	while bs < B:
		be = min(B, bs+hsic_compute_batch_size)
		dist_matrices[bs:be] = kernel_fn(dist_matrices[bs:be])
		bs = be
	kernel_matrices = dist_matrices.permute([2, 0, 1])
	assert list(kernel_matrices).shape == [B, K, K]

	self_hsic = []
	bs = 0
	bi = 0
	while bs < B:
		be = min(B, bs+hsic_compute_batch_size)
		self_hsic += [hsic.total_hsic_parallel(kernel_matrices[bs:be].unsqueeze(-1).repeat([1, 1, 1, 2]), return_log=True)]
		bs = be
		bi += 1
	self_hsic = torch.cat(self_hsic, dim=0)
	assert list(self_hsic.shape) == [B]

	winner_index = result.scores_B.argmax().item()

	ack_bag = [winner_index]
	acquisition_bag_scores = [0]
	batch_kernel = kernel_matrices[winner_index:winner_index+1].unsqueeze(-1) # n, n, cur_batch_size
	print('Computing HSIC for', B, 'points')
	for ackb_i in range(1, b):
		hsic_scores = []
		bs = 0
		bi = 0
		while bs < B:
			be = min(B, bs+hsic_compute_batch_size)
			m = be-bs
			hsic_scores += [hsic.total_hsic_parallel(
				torch.cat(
					[
						batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]), 
						kernel_matrices[bs:be].unsqueeze(-1)
					],
					dim=-1),
				return_log=True
				)
		hsic_scores = torch.cat(hsic_scores)
		assert hsic_scores.shape == self_hsic.shape
		hsic_scores -= self_hsic
		hsic_scores[ack_bag] = math.inf
		
		winner_index = hsic_scores.argmin().item()
		assert winner_index not in ack_bag
		winner_hsic = hsic_scores[winner_index].exp().item()

		print('hsic of batch', winner_hsic)

		batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_index].unsqueeze(-1)])
		ack_bag += [winner_index]
		acquisition_bag_scores += [winner_hsic]

    return AcquisitionBatch(ack_bag, acquisition_bag_scores, None)

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
                            joint_entropy_sampling.batch(probs_b_K_C.to(device), prev_samples_M_K), non_blocking=True
                        )

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

            print(f"Acquisition bag: {sorted(global_acquisition_bag)}")

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None)


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
