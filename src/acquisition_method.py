import enum

from torch import nn as nn

import independent_batch_acquisition
import multi_bald
from acquisition_batch import AcquisitionBatch
from acquisition_functions import AcquisitionFunction


class AcquisitionMethod(enum.Enum):
    independent = "independent"
    multibald = "multibald"
    hsicbald = "hsicbald"
    fass = "fass"

    def acquire_batch(
        self,
        bayesian_model: nn.Module,
        acquisition_function: AcquisitionFunction,
        available_loader,
        num_classes,
        k,
        b,
        min_candidates_per_acquired_item,
        min_remaining_percentage,
        initial_percentage,
        reduce_percentage,
        hsic_compute_batch_size=None,
        hsic_kernel_name=None,
        device=None,
    ) -> AcquisitionBatch:
        target_size = max(
            min_candidates_per_acquired_item * b, len(available_loader.dataset) * min_remaining_percentage // 100
        )

        if self == self.independent:
            return independent_batch_acquisition.compute_acquisition_bag(
                bayesian_model=bayesian_model,
                acquisition_function=acquisition_function,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                available_loader=available_loader,
                device=device,
            )
        elif self == self.multibald:
            return multi_bald.compute_multi_bald_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                device=device,
            )
        elif self == self.hsicbald:
            return multi_bald.compute_multi_hsic_batch4(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                hsic_compute_batch_size=hsic_compute_batch_size,
                hsic_kernel_name=hsic_kernel_name,
                device=device,
            )
        elif self == self.fass:
            return multi_bald.compute_fass_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                hsic_compute_batch_size=hsic_compute_batch_size,
                hsic_kernel_name=hsic_kernel_name,
                device=device,
            )
        else:
            raise NotImplementedError(f"Unknown acquisition method {self}!")
