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


class DDPBucketed(nn.Module):
    """
    A DDP implementation that uses gradient bucketing with computation-communication overlap.
    
    This implementation combines both key optimizations:
    1. Gradient bucketing: Groups parameters into buckets to reduce number of all-reduce calls
    2. Computation overlap: Triggers all-reduce as soon as a bucket is ready during backward pass
    
    This is similar to PyTorch's official DDP implementation.
    
    Args:
        module: The underlying model to wrap with DDP
        bucket_size_mb: Maximum size of each bucket in megabytes
    """
    
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Create buckets and register hooks
        self._create_buckets()
        self._register_gradient_hooks()
        
        # Track pending all-reduce operations
        self._pending_reductions: List[dist.Work] = []
        
        # Track which buckets have been reduced (for debugging)
        self._buckets_reduced = set()
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _create_buckets(self):
        """
        Create buckets of parameters in reverse order.
        
        Gradients become ready in approximately reverse order of parameters during
        backward pass, so we bucket in reverse order to trigger communication ASAP.
        """
        self.buckets = []  # List of buckets, each bucket is a list of parameters
        self.bucket_buffers = []  # Pre-allocated buffers for flattened gradients
        self.param_to_bucket = {}  # Map from parameter to (bucket_idx, offset_in_bucket, numel)
        
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024
        
        # Get parameters in reverse order
        params_with_grad = [p for p in self.module.parameters() if p.requires_grad]
        params_reversed = list(reversed(params_with_grad))
        
        current_bucket = []
        current_bucket_size = 0
        
        for param in params_reversed:
            param_size = param.numel() * param.element_size()
            
            # If adding this parameter would exceed bucket size and bucket is not empty,
            # start a new bucket
            if current_bucket_size + param_size > bucket_size_bytes and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append(param)
            current_bucket_size += param_size
        
        # Add the last bucket if not empty
        if current_bucket:
            self.buckets.append(current_bucket)
        
        # Pre-allocate buffers for each bucket and build param_to_bucket mapping
        for bucket_idx, bucket in enumerate(self.buckets):
            total_numel = sum(p.numel() for p in bucket)
            
            # Will allocate buffer on first use (need to know dtype and device)
            self.bucket_buffers.append(None)
            
            # Map each parameter to its bucket and offset
            offset = 0
            for param in bucket:
                self.param_to_bucket[param] = (bucket_idx, offset, param.numel())
                offset += param.numel()
        
        # Track which parameters in each bucket have gradients ready
        self._bucket_grads_ready = [set() for _ in self.buckets]
    
    def _register_gradient_hooks(self):
        """Register hooks to trigger bucket all-reduce when all grads in bucket are ready."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_bucket_hook(param))
    
    def _make_bucket_hook(self, param: nn.Parameter):
        """
        Create a hook that checks if a bucket is ready and triggers all-reduce.
        
        A bucket is ready when all of its parameters have gradients.
        """
        def hook(_param: nn.Parameter):
            if param.grad is None:
                return
            
            bucket_idx, offset, numel = self.param_to_bucket[param]
            
            # Mark this parameter's gradient as ready
            self._bucket_grads_ready[bucket_idx].add(param)
            
            # Check if all parameters in this bucket have gradients ready
            bucket = self.buckets[bucket_idx]
            if len(self._bucket_grads_ready[bucket_idx]) == len(bucket):
                # All gradients in this bucket are ready - trigger all-reduce
                self._all_reduce_bucket(bucket_idx)
        
        return hook
    
    def _all_reduce_bucket(self, bucket_idx: int):
        """
        All-reduce the gradients for a specific bucket.
        
        Flattens the gradients from all parameters in the bucket, issues an
        async all-reduce, and stores the handle.
        """
        if bucket_idx in self._buckets_reduced:
            # Already reduced this bucket (shouldn't happen, but be safe)
            return
        
        bucket = self.buckets[bucket_idx]
        
        # Allocate buffer if needed
        if self.bucket_buffers[bucket_idx] is None:
            total_numel = sum(p.numel() for p in bucket)
            # Use dtype and device from first parameter
            first_param = bucket[0]
            self.bucket_buffers[bucket_idx] = torch.empty(
                total_numel,
                dtype=first_param.grad.dtype,
                device=first_param.grad.device
            )
        
        buffer = self.bucket_buffers[bucket_idx]
        
        # Flatten gradients into buffer
        offset = 0
        for param in bucket:
            numel = param.grad.numel()
            buffer[offset:offset + numel].copy_(param.grad.view(-1))
            offset += numel
        
        # Async all-reduce
        handle = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_reductions.append((bucket_idx, handle))
        self._buckets_reduced.add(bucket_idx)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all pending gradient reductions and copy results back to parameters.
        
        This should be called after the backward pass is complete, but before
        the optimizer step.
        """
        world_size = dist.get_world_size()
        
        # Wait for all pending all-reduce operations
        for bucket_idx, handle in self._pending_reductions:
            handle.wait()
            
            # Copy reduced gradients back to parameters
            buffer = self.bucket_buffers[bucket_idx]
            buffer.div_(world_size)
            
            bucket = self.buckets[bucket_idx]
            offset = 0
            for param in bucket:
                numel = param.grad.numel()
                param.grad.copy_(buffer[offset:offset + numel].view_as(param.grad))
                offset += numel
        
        # Clear state for next iteration
        self._pending_reductions.clear()
        self._buckets_reduced.clear()
        for bucket_ready_set in self._bucket_grads_ready:
            bucket_ready_set.clear()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)
