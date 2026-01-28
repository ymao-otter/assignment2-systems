import logging
from copy import deepcopy
from typing import Type

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .adapters import (
    ddp_flattened_gradients_on_after_backward,
    get_ddp_flattened_gradients,
)
from .common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_DistributedDataParallelFlattenedGradients(model_class):
    world_size = 2
    mp.spawn(
        _test_DistributedDataParallelFlattenedGradients,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_DistributedDataParallelFlattenedGradients(
    rank: int, world_size: int, model_class: Type[nn.Module]
):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    dist.barrier()

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    # Create a toy model and move it to the proper device.
    # This is our non-parallel baseline.
    non_parallel_model = model_class().to(device)

    # Create a DDP model. Note that the weights of this model should
    # match the non-parallel baseline above.
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_flattened_gradients(ddp_base)

    # If we're on rank 0, the DDP model should still exactly match the parameters of the
    # non-parallel baseline (since the parameters on rank 0 weren't changed).
    # If we're not on rank 0, the DDP model's parameters should have been updated with
    # the parameters from rank 0. So, double-check that the parameter differ from the
    # local initial state.
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        # This parameter was initialized as [2, 2], so we expect its value to remain the same
        is_no_grad_fixed_param = (
            "no_grad_fixed_param" in ddp_model_param_name or "no_grad_fixed_param" in non_parallel_param_name
        )
        if rank == 0 or is_no_grad_fixed_param:
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(ddp_model)

    # Load the dataset from disk, so we can ensure that every rank has the same
    # overall pool of data.
    # Shape: (20, 10)
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    # Shape: (20, 5)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")

    # Each rank will see only 10 examples (out of the total dataset size of 20)
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    for i in range(5):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # Run the non-parallel model on all the data and take a gradient step
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # Run the DDP model on only the local shard of data
        local_x = all_x[rank * local_bs : (rank + 1) * local_bs].to(device)
        local_y = all_y[rank * local_bs : (rank + 1) * local_bs].to(device)
        ddp_output = ddp_model(local_x)
        ddp_loss = loss_fn(ddp_output, local_y)
        ddp_loss.backward()

        # Synchronize gradients before optimizer step
        ddp_flattened_gradients_on_after_backward(ddp_model, ddp_optimizer)

        ddp_optimizer.step()

        # Check that parameters match after training
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter, rtol=1e-4, atol=1e-5)

    _cleanup_process_group()
