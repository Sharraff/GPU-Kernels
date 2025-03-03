#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#define CUDA_CHECK(call)                                                                      \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess)                                                               \
        {                                                                                     \
            printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

__global__ void layer_norm_kernel(const float *input, float *output, const float *gamma, const float *beta, int N, int C, float epsilon)
{
    extern __shared__ float shared_data[]; // Shared memory for intermediate calculations

    int idx = blockIdx.x;  // Each block handles one sample in the batch
    int tid = threadIdx.x; // Thread ID within the block

    // Compute mean
    float mean = 0.0f;
    for (int j = tid; j < C; j += blockDim.x)
    {
        mean += input[idx * C + j];
    }
    shared_data[tid] = mean;
    __syncthreads();

    // Parallel reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    mean = shared_data[0] / C;

    // Compute variance
    float variance = 0.0f;
    for (int j = tid; j < C; j += blockDim.x)
    {
        float diff = input[idx * C + j] - mean;
        variance += diff * diff;
    }
    shared_data[tid] = variance;
    __syncthreads();

    // Parallel reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    variance = shared_data[0] / C;

    // Normalize and apply scale (gamma) and shift (beta)
    for (int j = tid; j < C; j += blockDim.x)
    {
        float normalized = (input[idx * C + j] - mean) / sqrtf(variance + epsilon);
        output[idx * C + j] = gamma[j] * normalized + beta[j];
    }
}

void layer_norm(const float *input, float *output, const float *gamma, const float *beta, int N, int C, float epsilon)
{
    float *d_input, *d_output, *d_gamma, *d_beta;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_input, N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_beta, C * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, input, N * C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, C * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threads_per_block = 256;                                // Adjust based on your hardware
    int shared_memory_size = threads_per_block * sizeof(float); // Shared memory size for reductions
    layer_norm_kernel<<<N, threads_per_block, shared_memory_size>>>(d_input, d_output, d_gamma, d_beta, N, C, epsilon);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, N * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

int main()
{
    int N = 10; // Batch size
    int C = 20; // Number of features
    float epsilon = 1e-5f;

    // Allocate host memory
    float *input = (float *)malloc(N * C * sizeof(float));
    float *output = (float *)malloc(N * C * sizeof(float));
    float *gamma = (float *)malloc(C * sizeof(float));
    float *beta = (float *)malloc(C * sizeof(float));

    // Initialize input, gamma, and beta with some values
    for (int i = 0; i < N * C; ++i)
    {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int j = 0; j < C; ++j)
    {
        gamma[j] = 1.0f;
        beta[j] = 0.0f;
    }

    // Perform layer normalization
    layer_norm(input, output, gamma, beta, N, C, epsilon);

    // Print the output
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < C; ++j)
        {
            printf("%f ", output[i * C + j]);
        }
        printf("\n");
    }

    // Free host memory
    free(input);
    free(output);
    free(gamma);
    free(beta);

    return 0;
}