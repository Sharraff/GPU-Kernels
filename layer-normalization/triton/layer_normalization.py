import triton
import triton.language as tl
import torch

class TritonLayerNorm:
    def __init__(self, eps: float = 1e-5):
        """
        Initialize the TritonLayerNorm class.

        Args:
            eps (float): Small constant for numerical stability. Default: 1e-5.
        """
        self.eps = eps

    @triton.jit
    def _layer_norm_kernel(
        x_ptr,
        y_ptr,
        gamma_ptr,
        beta_ptr,
        n_elements,
        eps,
        stride_x,
        stride_y,  
        BLOCK_SIZE: tl.constexpr,
    ):
        # Get the program ID (thread block index)
        pid = tl.program_id(axis=0)
        
        # Compute the base pointer for the input and output tensors
        x_offset = pid * stride_x
        y_offset = pid * stride_y
        
        # Create a range of indices for the current block
        row_idx = tl.arange(0, BLOCK_SIZE)
        
        # Load the input data into shared memory with memory coalescing (FP16)
        mask = row_idx < n_elements
        x = tl.load(x_ptr + x_offset + row_idx, mask=mask, other=0.0)
        
        # Convert input to FP32 for numerical stability during reduction
        x_f32 = x.to(tl.float32)
        
        # Compute mean using block-level reduction (FP32)
        mean = tl.sum(x_f32, axis=0) / n_elements
        
        # Compute variance using block-level reduction (FP32)
        diff = x_f32 - mean
        variance = tl.sum(tl.where(diff != 0, diff * diff, 0), axis=0) / n_elements
        
        # Normalize the input (FP32)
        x_normalized = (x_f32 - mean) / tl.sqrt(variance + eps)
        
        # Load the scale and shift parameters with memory coalescing (FP16)
        gamma = tl.load(gamma_ptr + row_idx, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + row_idx, mask=mask, other=0.0)
        
        # Convert scale and shift to FP32 for computation
        gamma_f32 = gamma.to(tl.float32)
        beta_f32 = beta.to(tl.float32)
        
        # Apply scale and shift (FP32)
        y_f32 = x_normalized * gamma_f32 + beta_f32
        
        # Convert the result back to FP16
        y = y_f32.to(tl.float16)
        
        # Store the result back to global memory with memory coalescing (FP16)
        tl.store(y_ptr + y_offset + row_idx, y, mask=mask)

    def forward(
        self,
        x: torch.Tensor,  
        gamma: torch.Tensor,  
        beta: torch.Tensor,
        stream: torch.cuda.Stream = None,
    ) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (n_batches, n_elements).
            gamma (torch.Tensor): Scale parameter of shape (n_elements,).
            beta (torch.Tensor): Shift parameter of shape (n_elements,).
            stream (torch.cuda.Stream): CUDA stream for asynchronous execution. Default: None.

        Returns:
            torch.Tensor: Normalized output tensor of the same shape as `x`.
        """
        # Ensure the input tensor is contiguous
        x = x.contiguous()
        
        # Get the shape of the input tensor
        n_batches, n_elements = x.shape
        
        # Allocate the output tensor (FP16)
        y = torch.empty_like(x)
        
        # Define the block size (must be a power of two)
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        
        # Launch the kernel asynchronously in the specified stream
        grid = (n_batches,)
        if stream is not None:
            with torch.cuda.stream(stream):
                self._layer_norm_kernel[grid](
                    x, y, gamma, beta, n_elements, self.eps, x.stride(0), y.stride(0), BLOCK_SIZE
                )
        else:
            self._layer_norm_kernel[grid](
                x, y, gamma, beta, n_elements, self.eps, x.stride(0), y.stride(0), BLOCK_SIZE
            )
        
        return y

# Example usage
if __name__ == "__main__":
    # Create random input tensor and parameters (FP16)
    x = torch.randn(32, 128, dtype=torch.float16).cuda()
    gamma = torch.ones(128, dtype=torch.float16).cuda()
    beta = torch.zeros(128, dtype=torch.float16).cuda()
    
    # Create an instance of TritonLayerNorm
    layer_norm = TritonLayerNorm(eps=1e-5)
    
    # Create a CUDA stream for asynchronous execution
    stream = torch.cuda.Stream()
    
    # Apply layer normalization asynchronously
    with torch.cuda.stream(stream):
        y = layer_norm.forward(x, gamma, beta, stream=stream)
    
    # Synchronize the stream to ensure the computation is complete
    stream.synchronize()
    
    print(y)