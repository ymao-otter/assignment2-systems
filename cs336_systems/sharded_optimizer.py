"""
Sharded Optimizer implementation for distributed training.

This module implements optimizer state sharding (similar to ZeRO-1), where each rank
only maintains optimizer state for a subset of parameters, reducing memory usage.
After each optimizer step, parameters are synchronized across all ranks.
"""

from typing import Any, Callable, Optional, Type, Iterator, Iterable
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    A sharded optimizer that distributes optimizer state across ranks.
    
    This optimizer wraps an arbitrary PyTorch optimizer and shards the parameters
    across all ranks in the distributed process group. Each rank only maintains
    optimizer state (e.g., momentum, variance) for its assigned parameters,
    reducing memory usage. After each optimizer step, updated parameters are
    synchronized across all ranks.
    
    Args:
        params: Collection of parameters to be optimized (or parameter groups)
        optimizer_cls: The optimizer class to wrap (e.g., torch.optim.AdamW)
        **kwargs: Additional keyword arguments forwarded to the optimizer constructor
    """
    
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        # Store the optimizer class and kwargs for later use
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        # Get distributed info
        if not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires distributed to be initialized")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Track which parameters belong to which rank
        self.param_to_rank = {}  # maps param -> rank that owns it
        self.owned_params = []  # parameters owned by this rank
        self.owned_param_groups = []  # param groups containing only owned params
        self.wrapped_optimizer = None
        self.has_params = False
        
        # Initialize the parent Optimizer class
        # Parent's __init__ will normalize params and call self.add_param_group() for each group
        super().__init__(params, {})
        
        # Create the wrapped optimizer with only the parameters owned by this rank
        if self.owned_param_groups:
            self.wrapped_optimizer = optimizer_cls(self.owned_param_groups, **kwargs)
            self.has_params = True
        else:
            # This rank doesn't own any parameters
            # We could create a dummy optimizer here, but it's simpler to just keep None
            # and check for it in step() and other methods
            pass
    
    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            **kwargs: Additional keyword arguments
            
        Returns:
            The loss if closure is provided, None otherwise
        """
        loss = None
        
        # Run the optimizer step on parameters owned by this rank
        if self.has_params:
            loss = self.wrapped_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            # Even if we don't own params, we might need to evaluate the closure
            loss = closure()
        
        # Synchronize updated parameters across all ranks
        # Each rank broadcasts its updated parameters to all other ranks
        self._synchronize_parameters()
        
        return loss
    
    def _synchronize_parameters(self):
        """
        Synchronize parameters across all ranks.
        
        After the optimizer step, each rank has updated its owned parameters.
        We need to broadcast these updated parameters to all other ranks so that
        all ranks have the same model parameters.
        """
        for param in self.param_to_rank.keys():
            assigned_rank = self.param_to_rank[param]
            # Broadcast this parameter from the rank that owns it
            dist.broadcast(param.data, src=assigned_rank)
    
    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a parameter group to the optimizer.
        
        This method is called by the parent Optimizer.__init__ during initialization.
        
        Args:
            param_group: A dict containing parameters and optimization options
        """
        # Let parent handle validation, normalization, duplicate checking, and adding to param_groups
        # Note: parent modifies param_group in place (normalizes params, adds defaults)
        super().add_param_group(param_group)
        
        # Use the normalized param_group (parent modified it in place)
        group_params = param_group['params']
        owned_params_in_group = []
        
        # Get current number of unique assigned parameters to continue round-robin
        current_param_count = len(self.param_to_rank)
        
        # Assign the new parameters to ranks
        for param in group_params:
            # Check if this parameter has already been assigned (tied weights)
            if param not in self.param_to_rank:
                # Continue round-robin assignment
                assigned_rank = current_param_count % self.world_size
                self.param_to_rank[param] = assigned_rank
                current_param_count += 1
                
                if assigned_rank == self.rank:
                    self.owned_params.append(param)
            
            # Add to owned params if this rank owns it
            if self.param_to_rank[param] == self.rank:
                owned_params_in_group.append(param)
        
        # Track owned param groups for later wrapped optimizer creation
        if owned_params_in_group:
            owned_group = {k: v for k, v in param_group.items() if k != 'params'}
            owned_group['params'] = owned_params_in_group
            self.owned_param_groups.append(owned_group)
    
    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        
        Note: This returns the state dict of the wrapped optimizer, which only
        contains state for parameters owned by this rank.
        """
        if self.has_params:
            return self.wrapped_optimizer.state_dict()
        else:
            return {'state': {}, 'param_groups': []}
    
    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        
        Args:
            state_dict: optimizer state dict (should be from the same rank)
        """
        if self.has_params:
            self.wrapped_optimizer.load_state_dict(state_dict)
