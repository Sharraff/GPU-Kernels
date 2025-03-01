import triton
import triton.language as tl
import torch

@triton.jit
def mean_kernel(
    x_ptr,
    mean_ptr,
    batch_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the current thread
    pid = tl.program_id(axis=0)

    # Calculate the range of elements this thread will handle
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < batch_size

    # Iterate over the hidden dimension
    for col_idx in range(hidden_size):
        # Load the input data
        x = tl.load(x_ptr + row_idx * hidden_size + col_idx, mask=mask)

        # Compute the sum of elements in this block
        block_sum = tl.sum(x, axis=0)

        # Atomically add the block sum to the global mean
        tl.atomic_add(mean_ptr + col_idx, block_sum)

    # Synchronize threads to ensure all blocks have finished
    tl.debug_barrier()

    # Normalize the global mean by the batch size
    if pid == 0:
        for col_idx in range(hidden_size):
            mean = tl.load(mean_ptr + col_idx) / batch_size
            tl.store(mean_ptr + col_idx, mean)

@triton.jit
def variance_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    batch_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the current thread
    pid = tl.program_id(axis=0)

    # Calculate the range of elements this thread will handle
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < batch_size

    # Iterate over the hidden dimension
    for col_idx in range(hidden_size):
        # Load the input data and the mean
        x = tl.load(x_ptr + row_idx * hidden_size + col_idx, mask=mask)
        mean = tl.load(mean_ptr + col_idx)

        # Compute the squared difference from the mean
        squared_diff = (x - mean) * (x - mean)

        # Compute the sum of squared differences in this block
        block_sum = tl.sum(squared_diff, axis=0)

        # Atomically add the block sum to the global variance
        tl.atomic_add(var_ptr + col_idx, block_sum)

    # Synchronize threads to ensure all blocks have finished
    tl.debug_barrier()

    # Normalize the global variance by the batch size
    if pid == 0:
        for col_idx in range(hidden_size):
            var = tl.load(var_ptr + col_idx) / batch_size
            tl.store(var_ptr + col_idx, var)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    eps,
    batch_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the current thread
    pid = tl.program_id(axis=0)

    # Calculate the range of elements this thread will handle
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < batch_size

    # Iterate over the hidden dimension
    for col_idx in range(hidden_size):
        # Load the input data, mean, and variance
        x = tl.load(x_ptr + row_idx * hidden_size + col_idx, mask=mask)
        mean = tl.load(mean_ptr + col_idx)
        var = tl.load(var_ptr + col_idx)

        # Load the gamma and beta parameters
        gamma = tl.load(gamma_ptr + col_idx)
        beta = tl.load(beta_ptr + col_idx)

        # Compute the normalized value
        normalized = (x - mean) / tl.sqrt(var + eps)

        # Apply the scale and shift
        y = gamma * normalized + beta

        # Store the result in global memory
        tl.store(y_ptr + row_idx * hidden_size + col_idx, y, mask=mask)

def batch_norm(x, gamma, beta, eps=1e-5):
    # Ensure the input is a contiguous tensor
    x = x.contiguous()
    
    # Get the shape of the input tensor
    batch_size, hidden_size = x.shape
    
    # Allocate the output tensor
    y = torch.empty_like(x)
    
    # Define the block size
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate tensors for mean and variance, initialized to zero
    mean = torch.zeros(hidden_size, device='cuda')
    var = torch.zeros(hidden_size, device='cuda')
    
    # Compute the global mean
    mean_kernel[(num_blocks,)](
        x_ptr=x, mean_ptr=mean, batch_size=batch_size,
        hidden_size=hidden_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute the global variance
    variance_kernel[(num_blocks,)](
        x_ptr=x, mean_ptr=mean, var_ptr=var, batch_size=batch_size,
        hidden_size=hidden_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Normalize the input using batch normalization
    batch_norm_kernel[(num_blocks,)](
        x_ptr=x, y_ptr=y, mean_ptr=mean, var_ptr=var,
        gamma_ptr=gamma, beta_ptr=beta, eps=eps, batch_size=batch_size,
        hidden_size=hidden_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

if __name__ == "__main__":
    # Create some dummy data
    batch_size, hidden_size = 128, 768
    x = torch.randn(batch_size, hidden_size, device='cuda')
    gamma = torch.randn(hidden_size, device='cuda')
    beta = torch.randn(hidden_size, device='cuda')

    # Apply batch normalization
    y = batch_norm(x, gamma, beta)

    # Print the result
    print(y)
