"""
Naive Distributed Data Parallel (DDP) implementation.

This module implements a basic DDP wrapper that all-reduces individual
parameter gradients after the backward pass.
"""

from typing import List
import torch
import torch.nn as nn
import torch.distributed as dist


class DDPIndividualParameters(nn.Module):
    """
    A naive DDP implementation that all-reduces gradients for each parameter individually.
    
    This implementation:
    1. Broadcasts parameters from rank 0 to all other ranks during initialization
    2. Registers post-accumulate gradient hooks to all-reduce gradients as they become ready
    3. Tracks pending gradient reduction operations for synchronization
    
    Args:
        module: The underlying model to wrap with DDP
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
        # List to track pending gradient reduction handles
        self._pending_grad_reductions: List[dist.Work] = []
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Register backward hooks for gradient synchronization
        self._register_gradient_hooks()
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            # Broadcast each parameter from rank 0
            dist.broadcast(param.data, src=0)
    
    def _register_gradient_hooks(self):
        """Register post-accumulate gradient hooks on all parameters that require gradients."""
        for param in self.module.parameters():
            if param.requires_grad:
                # Register a hook that will be called after the gradient is accumulated into param.grad
                param.register_post_accumulate_grad_hook(self._make_gradient_hook(param))
    
    def _make_gradient_hook(self, param: nn.Parameter):
        """
        Create a post-accumulate gradient hook for a specific parameter.
        
        This hook is called after the gradient has been accumulated into param.grad.
        
        Args:
            param: The parameter to create a hook for
            
        Returns:
            A hook function that all-reduces the gradient
        """
        def hook(_param: nn.Parameter):
            # At this point, param.grad is set and we can all-reduce it
            if param.grad is not None:
                # All-reduce the gradient asynchronously
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self._pending_grad_reductions.append(handle)
        
        return hook
    
    def finish_gradient_synchronization(self):
        """
        Wait for all pending gradient reductions to complete.
        
        This should be called after the backward pass is complete, but before
        the optimizer step.
        """
        # Wait for all pending all-reduce operations to complete
        for handle in self._pending_grad_reductions:
            handle.wait()
        
        # Average the gradients by dividing by world size
        # This is necessary because each rank computes gradients with respect to
        # the mean loss over its local batch. When we sum these gradients, we get
        # gradients that are world_size times larger than if we had processed all
        # data together in one batch.
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.div_(world_size)
        
        # Clear the list of pending reductions for the next iteration
        self._pending_grad_reductions.clear()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)
