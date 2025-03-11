import torch
import triton
import triton.language as tl

class TritonSoftmax:
    def __init__(self, BLOCK_SIZE=1024, num_warps=4):
        """
        Initialize the TritonSoftmax class.

        Args:
            BLOCK_SIZE (int): Block size for the kernel. Must be a power of two.
            num_warps (int): Number of warps per block.
        """
        self.BLOCK_SIZE = BLOCK_SIZE
        self.num_warps = num_warps

    @triton.jit
    def softmax_kernel(
        self,
        output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, 
        BLOCK_SIZE: tl.constexpr
    ):
        # Softmax is computed row-wise, so each program processes one row.
        row_idx = tl.program_id(0)
        
        # Pointers to the row in the input and output matrices.
        row_start_ptr = input_ptr + row_idx * input_row_stride
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        
        # Allocate shared memory for the row and intermediate results.
        row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        max_val = tl.zeros((1,), dtype=tl.float32)
        exp_sum = tl.zeros((1,), dtype=tl.float32)
        
        # Load the row into shared memory.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        
        # Block-level reduction to compute the maximum value in the row.
        max_val = tl.max(row, axis=0)
        
        # Subtract the maximum value for numerical stability.
        row_minus_max = row - max_val
        
        # Compute the exponential of each element.
        exp_row = tl.exp(row_minus_max)
        
        # Block-level reduction to compute the sum of exponentials.
        exp_sum = tl.sum(exp_row, axis=0)
        
        # Normalize by the sum of exponentials.
        softmax_output = exp_row / exp_sum
        
        # Write the result back to the output matrix.
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    def __call__(self, x: torch.Tensor, stream=None):
        """
        Compute the Softmax of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (n_rows, n_cols).
            stream (torch.cuda.Stream, optional): CUDA stream for asynchronous execution.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        n_rows, n_cols = x.shape
        
        # Allocate output tensor.
        output = torch.empty_like(x)
        
        # Define the block size. This should be a power of two and less than or equal to the number of columns.
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(BLOCK_SIZE, self.BLOCK_SIZE)  # Limit block size to the specified value.
        
        # Launch the kernel with CUDA streams for asynchronous execution.
        grid = (n_rows,)  # One block per row.
        
        if stream is not None:
            with torch.cuda.stream(stream):
                self.softmax_kernel[grid](
                    output, x, x.stride(0), output.stride(0), n_cols, 
                    BLOCK_SIZE=BLOCK_SIZE, num_warps=self.num_warps
                )
        else:
            self.softmax_kernel[grid](
                output, x, x.stride(0), output.stride(0), n_cols, 
                BLOCK_SIZE=BLOCK_SIZE, num_warps=self.num_warps
            )
        
        return output

# Example usage
if __name__ == "__main__":
    # Create an instance of the TritonSoftmax class
    softmax_op = TritonSoftmax(BLOCK_SIZE=1024, num_warps=4)

    # Input tensor
    x = torch.randn(1000, 256, device='cuda')

    # Compute Softmax
    output = softmax_op(x)

    print(output)