#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// CUDA kernel to compute mean of each feature using registers and shared memory
__global__ void mean_kernel(const float *input, float *mean, int batch_size, int seq_length)
{
    extern __shared__ float shared_mem[];
    int feature_idx = blockIdx.x; // Each block handles one feature (sequence position)
    int tid = threadIdx.x;
    float sum = 0.0f; // Use a register for the partial sum

    // Each thread computes a partial sum for its assigned batch elements
    for (int i = tid; i < batch_size; i += blockDim.x)
    {
        int idx = i * seq_length + feature_idx;
        sum += input[idx]; // Accumulate in a register
    }

    // Store partial sum in shared memory
    shared_mem[tid] = sum;
    __syncthreads();

    // Perform block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_mem[tid] += shared_mem[tid + stride]; // Accumulate in shared memory
        }
        __syncthreads();
    }

    // Write the mean to global memory
    if (tid == 0)
    {
        mean[feature_idx] = shared_mem[0] / batch_size;
        printf("Mean[%d] = %f\n", feature_idx, mean[feature_idx]);
    }
}

// CUDA kernel to compute variance of each feature using registers and shared memory
__global__ void variance_kernel(const float *input, const float *mean, float *variance, int batch_size, int seq_length)
{
    extern __shared__ float shared_mem[];
    int feature_idx = blockIdx.x; // Each block handles one feature (sequence position)
    int tid = threadIdx.x;
    float sum = 0.0f; // Use a register for the partial sum

    // Each thread computes a partial sum of squared differences
    for (int i = tid; i < batch_size; i += blockDim.x)
    {
        int idx = i * seq_length + feature_idx;
        float diff = input[idx] - mean[feature_idx];
        sum += diff * diff; // Accumulate in a register
    }

    // Store partial sum in shared memory
    shared_mem[tid] = sum;
    __syncthreads();

    // Perform block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_mem[tid] += shared_mem[tid + stride]; // Accumulate in shared memory
        }
        __syncthreads();
    }

    // Write the variance to global memory
    if (tid == 0)
    {
        variance[feature_idx] = shared_mem[0] / batch_size;
        printf("Variance[%d] = %f\n", feature_idx, variance[feature_idx]);
    }
}

// CUDA kernel to apply batch normalization for 2D tensors
__global__ void apply_batch_norm(float *input, const float *mean, const float *variance, const float *gamma, const float *beta, float epsilon, int batch_size, int seq_length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_length)
    {
        int batch_idx = idx / seq_length;
        int feature_idx = idx % seq_length;

        float normalized = (input[idx] - mean[feature_idx]) / sqrtf(variance[feature_idx] + epsilon);
        input[idx] = gamma[feature_idx] * normalized + beta[feature_idx];

        // Print normalized value for each element
        printf("Normalized[%d, %d] = %f\n", batch_idx, feature_idx, input[idx]);
    }
}

// Host function to perform batch normalization for 2D tensors
void batch_norm(float *d_input, int batch_size, int seq_length, float *d_gamma, float *d_beta, float epsilon)
{
    float *d_mean, *d_variance;
    cudaMalloc((void **)&d_mean, seq_length * sizeof(float));
    cudaMalloc((void **)&d_variance, seq_length * sizeof(float));

    // Compute mean
    printf("Computing mean...\n");
    mean_kernel<<<seq_length, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_mean, batch_size, seq_length);
    cudaDeviceSynchronize(); // Ensure kernel completes before proceeding

    // Compute variance
    printf("Computing variance...\n");
    variance_kernel<<<seq_length, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_mean, d_variance, batch_size, seq_length);
    cudaDeviceSynchronize(); // Ensure kernel completes before proceeding

    // Apply batch normalization
    printf("Applying batch normalization...\n");
    int total_elements = batch_size * seq_length;
    apply_batch_norm<<<(total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_mean, d_variance, d_gamma, d_beta, epsilon, batch_size, seq_length);
    cudaDeviceSynchronize(); // Ensure kernel completes before proceeding

    // Free memory
    cudaFree(d_mean);
    cudaFree(d_variance);
}

int main()
{
    // Example usage for 2D tensor (text data)
    int batch_size = 4; // Number of sequences in the batch
    int seq_length = 8; // Length of each sequence
    float epsilon = 1e-5f;

    // Allocate and initialize memory on host
    float *h_input = (float *)malloc(batch_size * seq_length * sizeof(float));
    float *h_gamma = (float *)malloc(seq_length * sizeof(float));
    float *h_beta = (float *)malloc(seq_length * sizeof(float));

    // Initialize input, gamma, and beta with some values
    for (int i = 0; i < batch_size * seq_length; ++i)
    {
        h_input[i] = 1.0f; // Example: all ones
    }
    for (int i = 0; i < seq_length; ++i)
    {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    // Print initial input values
    printf("Initial input values:\n");
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < seq_length; ++j)
        {
            printf("h_input[%d, %d] = %f\n", i, j, h_input[i * seq_length + j]);
        }
    }

    // Allocate memory on device
    float *d_input, *d_gamma, *d_beta;
    cudaMalloc((void **)&d_input, batch_size * seq_length * sizeof(float));
    cudaMalloc((void **)&d_gamma, seq_length * sizeof(float));
    cudaMalloc((void **)&d_beta, seq_length * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, batch_size * seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, seq_length * sizeof(float), cudaMemcpyHostToDevice);

    // Perform batch normalization
    batch_norm(d_input, batch_size, seq_length, d_gamma, d_beta, epsilon);

    // Copy result back to host
    cudaMemcpy(h_input, d_input, batch_size * seq_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Print normalized output values
    printf("Normalized output values:\n");
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < seq_length; ++j)
        {
            printf("h_input[%d, %d] = %f\n", i, j, h_input[i * seq_length + j]);
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    // Free host memory
    free(h_input);
    free(h_gamma);
    free(h_beta);

    return 0;
}