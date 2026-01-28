"""
Distributed Data Parallel (DDP) implementations.

This module implements several DDP variants with different communication strategies.
"""

from typing import List
import torch
import torch.nn as nn
import torch.distributed as dist


class DDPIndividualParameters(nn.Module):
    """
    The most naive DDP implementation that all-reduces each parameter synchronously.
    
    This implementation:
    1. Broadcasts parameters from rank 0 to all other ranks during initialization
    2. After backward pass, loops through each parameter and all-reduces synchronously
    3. Waits for all backward pass to complete before starting any communication
    
    This is the baseline for comparison - it conducts a separate all-reduce operation
    for every parameter tensor and waits for the backward pass to finish before
    communicating gradients.
    
    Args:
        module: The underlying model to wrap with DDP
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            # Broadcast each parameter from rank 0
            dist.broadcast(param.data, src=0)
    
    def finish_gradient_synchronization(self):
        """
        Synchronously all-reduce each parameter gradient.
        
        This should be called after the backward pass is complete, but before
        the optimizer step.
        """
        world_size = dist.get_world_size()
        
        # Loop through each parameter and all-reduce synchronously
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                # Synchronous all-reduce for this parameter
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                # Average the gradient
                param.grad.div_(world_size)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)


class DDPIndividualParametersOverlapped(nn.Module):
    """
    A DDP implementation that overlaps communication with computation.
    
    This implementation:
    1. Broadcasts parameters from rank 0 to all other ranks during initialization
    2. Registers post-accumulate gradient hooks to all-reduce gradients as they become ready
    3. Tracks pending gradient reduction operations for synchronization
    
    This allows communication of gradients to overlap with computation of the backward pass,
    but still issues one all-reduce per parameter.
    
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
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.div_(world_size)
        
        # Clear the list of pending reductions for the next iteration
        self._pending_grad_reductions.clear()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)


class DDPFlattenedGradients(nn.Module):
    """
    A DDP implementation that flattens all gradients into a single tensor before all-reduce.
    
    This implementation reduces communication overhead by issuing a single all-reduce operation
    instead of one per parameter.
    
    This implementation:
    1. Broadcasts parameters from rank 0 to all other ranks during initialization
    2. Flattens all gradients into a single contiguous tensor
    3. Performs a single all-reduce operation on the flattened gradient tensor
    4. Copies the reduced gradients back to individual parameter gradients
    
    Args:
        module: The underlying model to wrap with DDP
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Store parameter shapes and total size for flattening/unflattening
        self._param_shapes = []
        self._param_numel = []
        self._total_numel = 0
        
        for param in self.module.parameters():
            if param.requires_grad:
                self._param_shapes.append(param.shape)
                numel = param.numel()
                self._param_numel.append(numel)
                self._total_numel += numel
        
        # Pre-allocate buffer for flattened gradients
        self._flattened_grad_buffer = None
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            # Broadcast each parameter from rank 0
            dist.broadcast(param.data, src=0)
    
    def finish_gradient_synchronization(self):
        """
        Flatten all gradients, all-reduce them in a single operation, then unflatten.
        
        This should be called after the backward pass is complete, but before
        the optimizer step.
        """
        # Collect all parameters with gradients (in consistent order)
        params_with_grad = []
        total_numel = 0
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad.append(param)
                total_numel += param.grad.numel()
        
        if not params_with_grad:
            return
        
        # Allocate or reallocate buffer if needed
        if self._flattened_grad_buffer is None or self._flattened_grad_buffer.numel() < total_numel:
            # Determine device and dtype from first parameter
            first_param = params_with_grad[0]
            self._flattened_grad_buffer = torch.empty(
                total_numel,
                dtype=first_param.grad.dtype,
                device=first_param.grad.device
            )
        
        # Use only the needed portion of the buffer
        flattened_grads = self._flattened_grad_buffer[:total_numel]
        
        # Flatten all gradients into the buffer
        offset = 0
        for param in params_with_grad:
            numel = param.grad.numel()
            flattened_grads[offset:offset + numel].copy_(param.grad.view(-1))
            offset += numel
        
        # All-reduce the flattened gradients (in-place)
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)
        
        # Average the gradients by dividing by world size
        world_size = dist.get_world_size()
        flattened_grads.div_(world_size)
        
        # Copy the reduced gradients back to individual parameter gradients
        offset = 0
        for param in params_with_grad:
            numel = param.grad.numel()
            param.grad.copy_(flattened_grads[offset:offset + numel].view_as(param.grad))
            offset += numel
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)
