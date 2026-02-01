"""
FlashAttention-2 implementations in PyTorch and Triton.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@torch.compile
def _flash_backward_tile(Q_i, K_j, V_j, dO_i, L_i, D_i, scale, is_causal, q_start, k_start):
    """
    Compiled helper function for backward pass computation of a single query-key tile pair.
    
    Args:
        Q_i: Query tile, shape (q_tile_size, d)
        K_j: Key tile, shape (k_tile_size, d)
        V_j: Value tile, shape (k_tile_size, d)
        dO_i: Gradient of output tile, shape (q_tile_size, d)
        L_i: Logsumexp for query tile, shape (q_tile_size,)
        D_i: D vector for query tile, shape (q_tile_size,)
        scale: Attention scale factor
        is_causal: Whether to apply causal masking
        q_start: Starting index of query tile
        k_start: Starting index of key tile
        
    Returns:
        dQ_i_contrib: Contribution to dQ from this tile, shape (q_tile_size, d)
        dK_j: Gradient for key tile, shape (k_tile_size, d)
        dV_j: Gradient for value tile, shape (k_tile_size, d)
    """
    q_end = q_start + Q_i.shape[0]
    k_end = k_start + K_j.shape[0]
    
    # Recompute attention scores S_ij = Q_i @ K_j^T * scale
    S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
    
    # Apply causal masking if needed
    if is_causal:
        q_indices = torch.arange(q_start, q_end, device=Q_i.device)[:, None]
        k_indices = torch.arange(k_start, k_end, device=K_j.device)[None, :]
        causal_mask = q_indices >= k_indices
        S_ij = torch.where(causal_mask, S_ij, torch.tensor(-1e6, device=S_ij.device, dtype=S_ij.dtype))
    
    # Compute attention weights P_ij = exp(S_ij - L_i)
    P_ij = torch.exp(S_ij - L_i[:, None])
    
    # Compute dV_j (Equation 14): dV_j = P_ij^T @ dO_i
    dV_j = torch.matmul(P_ij.float().transpose(-2, -1), dO_i.float())
    
    # Compute dP_ij (Equation 15): dP_ij = dO_i @ V_j^T
    dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))
    
    # Compute dS_ij (Equation 16-19): dS_ij = P_ij * (dP_ij - D_i)
    dS_ij = P_ij * (dP_ij - D_i[:, None])
    
    # Apply causal masking to gradients if needed
    if is_causal:
        dS_ij = torch.where(causal_mask, dS_ij, torch.zeros_like(dS_ij))
    
    # Compute dQ_i contribution: dS_ij @ K_j * scale
    dQ_i_contrib = torch.matmul(dS_ij.float(), K_j.float()) * scale
    
    # Compute dK_j: dS_ij^T @ Q_i * scale
    dK_j = torch.matmul(dS_ij.float().transpose(-2, -1), Q_i.float()) * scale
    
    return dQ_i_contrib, dK_j, dV_j


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    Pure PyTorch implementation of FlashAttention-2 forward pass.
    
    This implements Algorithm 1 from the FlashAttention-2 paper using tiled computation.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass of FlashAttention-2.
        
        Args:
            ctx: Context object for saving tensors
            Q: Query tensor of shape (batch_size, n_queries, d)
            K: Key tensor of shape (batch_size, n_keys, d)
            V: Value tensor of shape (batch_size, n_keys, d)
            is_causal: Whether to apply causal masking (default: False)
            
        Returns:
            O: Output tensor of shape (batch_size, n_queries, d)
        """
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        
        # Scale factor
        scale = 1.0 / (d ** 0.5)
        
        # Tile sizes (must be at least 16x16)
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        
        # Number of tiles
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE
        
        # Initialize output and logsumexp
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, n_queries, device=Q.device, dtype=torch.float32)
        
        # Process each batch independently
        for b in range(batch_size):
            # Process each query tile
            for i in range(Tq):
                # Get query tile indices
                q_start = i * Q_TILE_SIZE
                q_end = min(q_start + Q_TILE_SIZE, n_queries)
                
                # Load query tile Q_i
                Q_i = Q[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                
                # Initialize on-chip buffers (in float32 for numerical stability)
                O_i = torch.zeros(q_end - q_start, d, device=Q.device, dtype=torch.float32)
                l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
                m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32)
                
                # Process each key tile
                for j in range(Tk):
                    # Get key tile indices
                    k_start = j * K_TILE_SIZE
                    k_end = min(k_start + K_TILE_SIZE, n_keys)
                    
                    # Load key and value tiles
                    K_j = K[b, k_start:k_end, :]  # Shape: (k_tile_size, d)
                    V_j = V[b, k_start:k_end, :]  # Shape: (k_tile_size, d)
                    
                    # Compute attention scores S_ij = Q_i @ K_j^T * scale
                    S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # Shape: (q_tile_size, k_tile_size)
                    
                    # Apply causal masking if needed
                    if is_causal:
                        # Create causal mask for this tile
                        q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                        k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                        causal_mask = q_indices >= k_indices
                        S_ij = torch.where(causal_mask, S_ij, torch.tensor(-1e6, device=S_ij.device, dtype=S_ij.dtype))
                    
                    # Compute row-wise max for numerical stability
                    m_ij = torch.max(S_ij, dim=-1).values  # Shape: (q_tile_size,)
                    
                    # Compute P_ij = exp(S_ij - m_ij) (tiled softmax, not globally normalized yet)
                    P_ij = torch.exp(S_ij - m_ij[:, None])  # Shape: (q_tile_size, k_tile_size)
                    
                    # Compute row sums for this tile
                    l_ij = torch.sum(P_ij, dim=-1)  # Shape: (q_tile_size,)
                    
                    # Update m_i
                    m_i_new = torch.maximum(m_i, m_ij)
                    
                    # Rescaling factors
                    exp_m_old = torch.exp(m_i - m_i_new)
                    exp_m_ij = torch.exp(m_ij - m_i_new)
                    
                    # Update l_i
                    l_i_new = exp_m_old * l_i + exp_m_ij * l_ij
                    
                    # Update O_i
                    # O_i = (l_i / l_i_new) * exp(m_i - m_i_new) * O_i + (1 / l_i_new) * exp(m_ij - m_i_new) * P_ij @ V_j
                    # Note: P_ij already has exp(m_ij) factored out, so we multiply by exp_m_ij
                    O_i = (exp_m_old * l_i / l_i_new)[:, None] * O_i + (exp_m_ij / l_i_new)[:, None] * torch.matmul(P_ij, V_j.float())
                    
                    # Update trackers
                    l_i = l_i_new
                    m_i = m_i_new
                
                # Write output tile back
                O[b, q_start:q_end, :] = O_i.to(Q.dtype)
                
                # Compute logsumexp: L_i = m_i + log(l_i)
                L[b, q_start:q_end] = m_i + torch.log(l_i)
        
        # Save tensors for backward pass
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass of FlashAttention-2.
        
        Implements equations 13-19 from FlashAttention-2 paper using tiled computation.
        
        Args:
            ctx: Context object with saved tensors
            dO: Gradient of output, shape (batch_size, n_queries, d)
            
        Returns:
            dQ, dK, dV: Gradients for Q, K, V
            None: Placeholder for is_causal gradient
        """
        # Load saved tensors
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        
        # Scale factor
        scale = 1.0 / (d ** 0.5)
        
        # Tile sizes (same as forward pass)
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        
        # Number of tiles
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE
        
        # Initialize gradients
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Compute D vector (Equation 13): D_i = rowsum(dO_i * O_i)
        D = torch.sum(dO * O, dim=-1)  # Shape: (batch_size, n_queries)
        
        # Process each batch independently
        for b in range(batch_size):
            # Process each query tile
            for i in range(Tq):
                # Get query tile indices
                q_start = i * Q_TILE_SIZE
                q_end = min(q_start + Q_TILE_SIZE, n_queries)
                
                # Load query tile and related tensors
                Q_i = Q[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                O_i = O[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                dO_i = dO[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                L_i = L[b, q_start:q_end]  # Shape: (q_tile_size,)
                D_i = D[b, q_start:q_end]  # Shape: (q_tile_size,)
                
                # Initialize gradient accumulator for this query tile
                dQ_i = torch.zeros_like(Q_i, dtype=torch.float32)
                
                # Process each key tile
                for j in range(Tk):
                    # Get key tile indices
                    k_start = j * K_TILE_SIZE
                    k_end = min(k_start + K_TILE_SIZE, n_keys)
                    
                    # Load key and value tiles
                    K_j = K[b, k_start:k_end, :]  # Shape: (k_tile_size, d)
                    V_j = V[b, k_start:k_end, :]  # Shape: (k_tile_size, d)
                    
                    # Use compiled helper for tile computation
                    dQ_contrib, dK_j, dV_j = _flash_backward_tile(
                        Q_i, K_j, V_j, dO_i, L_i, D_i, scale, is_causal, q_start, k_start
                    )
                    
                    # Accumulate gradients
                    dQ_i += dQ_contrib
                    dK[b, k_start:k_end, :] += dK_j.to(dK.dtype)
                    dV[b, k_start:k_end, :] += dV_j.to(dV.dtype)
                
                # Write dQ_i back
                dQ[b, q_start:q_end, :] = dQ_i.to(dQ.dtype)
        
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Pointer for L (logsumexp)
    L_offset = batch_index * stride_lb + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) * stride_lq
    
    # Load query tile
    Q_tile = tl.load(Q_block_ptr)  # Shape: (Q_TILE_SIZE, D)
    
    # Initialize accumulators (on-chip buffers in float32)
    O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], value=-float('inf'), dtype=tl.float32)
    
    # Number of key tiles
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    # Query indices for causal masking
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)
    
    # Loop over key tiles
    for j in range(Tk):
        # Load key and value tiles
        K_tile = tl.load(K_block_ptr)  # Shape: (D, K_TILE_SIZE)
        V_tile = tl.load(V_block_ptr)  # Shape: (K_TILE_SIZE, D)
        
        # Compute attention scores: S_ij = Q_i @ K_j^T * scale
        S_ij = tl.dot(Q_tile, K_tile) * scale  # Shape: (Q_TILE_SIZE, K_TILE_SIZE)
        
        # Apply causal masking if needed
        if is_causal:
            k_start = j * K_TILE_SIZE
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, -1e6)
        
        # Compute row-wise max for numerical stability
        m_ij = tl.max(S_ij, axis=1)  # Shape: (Q_TILE_SIZE,)
        
        # Compute P_ij = exp(S_ij - m_ij)
        P_ij = tl.exp(S_ij - m_ij[:, None])  # Shape: (Q_TILE_SIZE, K_TILE_SIZE)
        
        # Compute row sums
        l_ij = tl.sum(P_ij, axis=1)  # Shape: (Q_TILE_SIZE,)
        
        # Update m_i
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Rescaling factors
        exp_m_old = tl.exp(m_i - m_i_new)
        exp_m_ij = tl.exp(m_ij - m_i_new)
        
        # Update l_i
        l_i_new = exp_m_old * l_i + exp_m_ij * l_ij
        
        # Update O_i
        # O_i = (exp_m_old * l_i / l_i_new) * O_i + (exp_m_ij / l_i_new) * P_ij @ V_j
        # Cast P_ij to V's dtype before multiplication
        P_ij = P_ij.to(V_tile.dtype)
        
        # Compute the new contribution
        PV = tl.dot(P_ij, V_tile)
        
        # Update O_i
        O_i = (exp_m_old * l_i / l_i_new)[:, None] * O_i + (exp_m_ij / l_i_new)[:, None] * PV
        
        # Update trackers
        l_i = l_i_new
        m_i = m_i_new
        
        # Advance block pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Cast O_i to output dtype and store
    O_i = O_i.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i)
    
    # Compute and store logsumexp: L_i = m_i + log(l_i)
    L_i = m_i + tl.log(l_i)
    tl.store(L_ptr + L_offset, L_i)


class FlashAttentionTriton(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2 forward pass.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass of FlashAttention-2 using Triton kernel.
        
        Args:
            ctx: Context object for saving tensors
            Q: Query tensor of shape (batch_size, n_queries, d)
            K: Key tensor of shape (batch_size, n_keys, d)
            V: Value tensor of shape (batch_size, n_keys, d)
            is_causal: Whether to apply causal masking (default: False)
            
        Returns:
            O: Output tensor of shape (batch_size, n_queries, d)
        """
        # Check that tensors are on CUDA
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA for Triton kernels"
        
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        
        # Scale factor
        scale = 1.0 / (d ** 0.5)
        
        # Tile sizes (tunable)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        
        # Initialize output and logsumexp
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=torch.float32)
        
        # Launch grid: (Tq, batch_size)
        Tq = triton.cdiv(n_queries, Q_TILE_SIZE)
        grid = (Tq, batch_size)
        
        # Launch kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        # Save tensors for backward pass
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass of FlashAttention-2.
        
        Implements equations 13-19 from FlashAttention-2 paper using tiled computation.
        This uses PyTorch operations (not Triton kernels) for the backward pass.
        
        Args:
            ctx: Context object with saved tensors
            dO: Gradient of output, shape (batch_size, n_queries, d)
            
        Returns:
            dQ, dK, dV: Gradients for Q, K, V
            None: Placeholder for is_causal gradient
        """
        # Load saved tensors
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]
        
        # Scale factor
        scale = 1.0 / (d ** 0.5)
        
        # Tile sizes (same as forward pass)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        
        # Number of tiles
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE
        
        # Initialize gradients
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Compute D vector (Equation 13): D_i = rowsum(dO_i * O_i)
        D = torch.sum(dO * O, dim=-1)  # Shape: (batch_size, n_queries)
        
        # Process each batch independently
        for b in range(batch_size):
            # Process each query tile
            for i in range(Tq):
                # Get query tile indices
                q_start = i * Q_TILE_SIZE
                q_end = min(q_start + Q_TILE_SIZE, n_queries)
                
                # Load query tile and related tensors
                Q_i = Q[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                O_i = O[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                dO_i = dO[b, q_start:q_end, :]  # Shape: (q_tile_size, d)
                L_i = L[b, q_start:q_end]  # Shape: (q_tile_size,)
                D_i = D[b, q_start:q_end]  # Shape: (q_tile_size,)
                
                # Initialize gradient accumulator for this query tile
                dQ_i = torch.zeros_like(Q_i, dtype=torch.float32)
                
                # Process each key tile
                for j in range(Tk):
                    # Get key tile indices
                    k_start = j * K_TILE_SIZE
                    k_end = min(k_start + K_TILE_SIZE, n_keys)
                    
                    # Load key and value tiles
                    K_j = K[b, k_start:k_end, :]  # Shape: (k_tile_size, d)
                    V_j = V[b, k_start:k_end, :]  # Shape: (k_tile_size, d)
                    
                    # Use compiled helper for tile computation
                    dQ_contrib, dK_j, dV_j = _flash_backward_tile(
                        Q_i, K_j, V_j, dO_i, L_i, D_i, scale, is_causal, q_start, k_start
                    )
                    
                    # Accumulate gradients
                    dQ_i += dQ_contrib
                    dK[b, k_start:k_end, :] += dK_j.to(dK.dtype)
                    dV[b, k_start:k_end, :] += dV_j.to(dV.dtype)
                
                # Write dQ_i back
                dQ[b, q_start:q_end, :] = dQ_i.to(dQ.dtype)
        
        return dQ, dK, dV, None
