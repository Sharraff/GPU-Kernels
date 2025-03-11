import torch
import triton
import triton.language as tl

class CrossEntropyLoss:
    def __init__(self, block_size: int = 128, tile_size: int = 32):
        """
        Initialize the CrossEntropyLoss class.

        Args:
            block_size (int): Number of threads per block. Default is 128.
            tile_size (int): Tile size for tiling over classes. Default is 32.
        """
        self.block_size = block_size
        self.tile_size = tile_size

    @triton.jit
    def _cross_entropy_kernel(
        logits_ptr,  # Pointer to the logits tensor
        targets_ptr, # Pointer to the targets tensor
        output_ptr,  # Pointer to the output tensor (loss)
        n_classes,   # Number of classes
        n_elements,  # Total number of elements in the batch
        BLOCK_SIZE: tl.constexpr,  # Block size for Triton
        TILE_SIZE: tl.constexpr,   # Tile size for tiling over classes
    ):
        # Get the program ID (which element in the batch we're processing)
        pid = tl.program_id(axis=0)
        
        # Create a range of indices for the current block
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Mask to avoid out-of-bounds access
        mask = offsets < n_elements
        
        # Load targets for the current block
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0)
        
        # Allocate shared memory for logits and intermediate results
        logits_shared = tl.zeros((BLOCK_SIZE, TILE_SIZE), dtype=tl.float32)
        max_logits_shared = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        sum_exp_shared = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        loss_shared = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        
        # Tile over the classes dimension
        for tile_start in range(0, n_classes, TILE_SIZE):
            tile_offsets = tile_start + tl.arange(0, TILE_SIZE)
            tile_mask = tile_offsets < n_classes
            
            # Load a tile of logits into shared memory
            logits_tile = tl.load(
                logits_ptr + offsets[:, None] * n_classes + tile_offsets[None, :],
                mask=mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            logits_shared = tl.where(
                tile_mask[None, :], logits_tile, logits_shared
            )
            
            # Compute max logits for numerical stability
            max_logits_tile = tl.max(logits_shared, axis=1)
            max_logits_shared = tl.maximum(max_logits_shared, max_logits_tile)
            
            # Compute exp(logits - max_logits) and sum for softmax
            logits_exp = tl.exp(logits_shared - max_logits_shared[:, None])
            sum_exp_shared += tl.sum(logits_exp, axis=1)
            
            # Compute log softmax for the target class
            target_mask = tile_offsets == targets[:, None]
            target_log_probs = tl.where(
                target_mask, tl.log(logits_exp / sum_exp_shared[:, None]), 0.0
            )
            loss_shared += tl.sum(target_log_probs, axis=1)
        
        # Block-level reduction for the loss
        loss = -tl.sum(loss_shared)
        
        # Store the result asynchronously
        tl.store(output_ptr + pid, loss, mask=mask, eviction_policy="evict_last")

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, n_classes).
            targets (torch.Tensor): Target class indices of shape (batch_size,).

        Returns:
            torch.Tensor: Cross-entropy loss for each element in the batch.
        """
        assert logits.is_cuda and targets.is_cuda, "Inputs must be CUDA tensors"
        assert logits.shape[0] == targets.shape[0], "Batch sizes must match"
        
        n_elements = logits.shape[0]
        n_classes = logits.shape[1]
        
        # Allocate output tensor
        output = torch.empty(n_elements, device=logits.device, dtype=logits.dtype)
        
        # Compute grid size
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Create a CUDA stream for asynchronous execution
        stream = torch.cuda.Stream()
        
        # Launch the kernel asynchronously
        with torch.cuda.stream(stream):
            self._cross_entropy_kernel[grid](
                logits, targets, output, n_classes, n_elements,
                BLOCK_SIZE=self.block_size, TILE_SIZE=self.tile_size
            )
        
        # Synchronize the stream to ensure the kernel completes
        stream.synchronize()
        
        return output

# Example usage
if __name__ == "__main__":
    # Create an instance of the CrossEntropyLoss class
    cross_entropy_loss = CrossEntropyLoss(block_size=128, tile_size=32)
    
    # Example inputs
    logits = torch.randn(1024, 10, device='cuda')  # Example logits
    targets = torch.randint(0, 10, (1024,), device='cuda')  # Example targets
    
    # Compute the loss
    loss = cross_entropy_loss(logits, targets)
    print(loss)  # Output: tensor([2.3456, 1.2345, ..., 3.4567], device='cuda:0')