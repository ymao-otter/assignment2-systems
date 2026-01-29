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
        # We need to handle params which could be either an iterable of tensors
        # or an iterable of dicts (param groups)
        
        # First, collect all param groups
        if isinstance(params, torch.Tensor):
            raise TypeError("params should be an iterable of Tensors or dicts")
        
        # Convert params to a list to process them
        params_list = list(params)
        
        if len(params_list) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        # Check if we have param groups (dicts) or just tensors
        if isinstance(params_list[0], dict):
            param_groups = params_list
        else:
            # Just tensors, create a single param group
            param_groups = [{'params': params_list}]
        
        # Store the optimizer class and kwargs for later use
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        # Get distributed info
        if not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires distributed to be initialized")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # We'll build our own param_groups that will be passed to the parent Optimizer.__init__
        # These will contain ALL parameters (not just the sharded ones)
        self.all_param_groups = []
        
        # Track which parameters belong to which rank
        self.param_to_rank = {}  # maps param -> rank that owns it
        self.owned_params = []  # parameters owned by this rank
        self.owned_param_groups = []  # param groups containing only owned params
        
        # Flag to track if we're in initialization (before wrapped_optimizer is created)
        self._initializing = True
        self.wrapped_optimizer = None
        self.has_params = False
        
        # Global counter for parameter assignment (to handle multiple param groups)
        global_param_idx = 0
        
        # Process each param group and assign parameters to ranks
        for group in param_groups:
            # Make a copy of the group dict
            group_copy = {k: v for k, v in group.items() if k != 'params'}
            group_params = list(group['params'])
            
            # Add to all_param_groups
            self.all_param_groups.append({**group_copy, 'params': group_params})
            
            # Shard the parameters in this group across ranks
            owned_params_in_group = []
            for param in group_params:
                # Check if this parameter has already been assigned (tied weights)
                if param not in self.param_to_rank:
                    # Assign parameter to a rank in round-robin fashion
                    assigned_rank = global_param_idx % self.world_size
                    self.param_to_rank[param] = assigned_rank
                    global_param_idx += 1
                    
                    if assigned_rank == self.rank:
                        self.owned_params.append(param)
                
                # Add to owned params if this rank owns it
                if self.param_to_rank[param] == self.rank:
                    owned_params_in_group.append(param)
            
            # Create param group for owned params
            if owned_params_in_group:
                self.owned_param_groups.append({**group_copy, 'params': owned_params_in_group})
        
        # Initialize the parent Optimizer class with ALL parameters
        # This is required so that .param_groups, .state, etc. are properly initialized
        super().__init__(self.all_param_groups, {})
        
        # Now that parent is initialized, create the wrapped optimizer
        self._initializing = False
        
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
        
        This method is called by the parent Optimizer.__init__ and may also be
        called during training (e.g., for gradually unfreezing layers).
        
        Args:
            param_group: A dict containing parameters and optimization options
        """
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dict")
        
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)
        
        # Check for duplicate parameters
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")
        
        # Assign the new parameters to ranks
        group_params = param_group['params']
        owned_params_in_group = []
        
        # Get current number of unique assigned parameters to continue round-robin
        current_param_count = len(self.param_to_rank)
        
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
        
        # Add to parent's param_groups
        super().add_param_group(param_group)
        
        # Add owned params to the wrapped optimizer (only if it exists, i.e., not during __init__)
        if owned_params_in_group and not self._initializing:
            # Create param group for wrapped optimizer with only owned params
            owned_group = {k: v for k, v in param_group.items() if k != 'params'}
            owned_group['params'] = owned_params_in_group
            self.wrapped_optimizer.add_param_group(owned_group)
            self.has_params = True
    
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
