#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 32 // Tile size for shared memory

__global__ void cross_entropy_kernel(const float *__restrict__ logits, const int *__restrict__ labels, float *losses, int num_classes, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Shared memory for storing intermediate results
    __shared__ float shared_max_logit[TILE_SIZE];
    __shared__ float shared_log_sum_exp[TILE_SIZE];
    __shared__ float shared_loss[TILE_SIZE];

    float thread_loss = 0.0f;

    if (idx < num_elements)
    {
        int label = labels[idx];

        // Process data in tiles
        for (int t = 0; t < num_classes; t += TILE_SIZE)
        {
            int tile_end = min(t + TILE_SIZE, num_classes);

            // Load logits into shared memory
            if (tid < tile_end - t)
            {
                shared_max_logit[tid] = logits[idx * num_classes + t + tid];
            }
            __syncthreads();

            // Find max logit in the tile
            float max_logit = shared_max_logit[0];
            for (int i = 1; i < tile_end - t; ++i)
            {
                if (shared_max_logit[i] > max_logit)
                {
                    max_logit = shared_max_logit[i];
                }
            }
            __syncthreads();

            // Compute log-sum-exp for the tile
            float log_sum_exp = 0.0f;
            for (int i = 0; i < tile_end - t; ++i)
            {
                log_sum_exp += expf(shared_max_logit[i] - max_logit);
            }
            log_sum_exp = logf(log_sum_exp) + max_logit;
            shared_log_sum_exp[tid] = log_sum_exp;
            __syncthreads();

            // Compute cross-entropy loss for the tile
            if (t <= label && label < tile_end)
            {
                thread_loss += shared_log_sum_exp[label - t] - logits[idx * num_classes + label];
            }
        }

        // Store thread-local loss in shared memory
        shared_loss[tid] = thread_loss;
        __syncthreads();

        // Block-level reduction to sum losses within the block
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tid < stride)
            {
                shared_loss[tid] += shared_loss[tid + stride];
            }
            __syncthreads();
        }

        // Write the block's total loss to global memory
        if (tid == 0)
        {
            losses[blockIdx.x] = shared_loss[0];
        }
    }
}

void cross_entropy_loss(const float *logits, const int *labels, float *losses, int num_classes, int num_elements)
{
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_entropy_kernel<<<num_blocks, BLOCK_SIZE>>>(logits, labels, losses, num_classes, num_elements);
}

int main()
{
    // Example usage
    int num_elements = 1000;
    int num_classes = 10;
    size_t logits_size = num_elements * num_classes * sizeof(float);
    size_t labels_size = num_elements * sizeof(int);
    size_t losses_size = num_elements * sizeof(float);

    float *h_logits = (float *)malloc(logits_size);
    int *h_labels = (int *)malloc(labels_size);
    float *h_losses = (float *)malloc(losses_size);

    // Initialize logits and labels with some values
    for (int i = 0; i < num_elements; ++i)
    {
        h_labels[i] = i % num_classes; // Example labels
        for (int j = 0; j < num_classes; ++j)
        {
            h_logits[i * num_classes + j] = static_cast<float>(rand()) / RAND_MAX; // Random logits
        }
    }

    float *d_logits;
    int *d_labels;
    float *d_losses;

    cudaMalloc(&d_logits, logits_size);
    cudaMalloc(&d_labels, labels_size);
    cudaMalloc(&d_losses, losses_size);

    cudaMemcpy(d_logits, h_logits, logits_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, labels_size, cudaMemcpyHostToDevice);

    cross_entropy_loss(d_logits, d_labels, d_losses, num_classes, num_elements);

    cudaMemcpy(h_losses, d_losses, losses_size, cudaMemcpyDeviceToHost);

    // Sum the losses across all blocks
    float total_loss = 0.0f;
    for (int i = 0; i < (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
    {
        total_loss += h_losses[i];
    }
    printf("Total Cross-Entropy Loss: %f\n", total_loss);

    free(h_logits);
    free(h_labels);
    free(h_losses);
    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_losses);

    return 0;
}