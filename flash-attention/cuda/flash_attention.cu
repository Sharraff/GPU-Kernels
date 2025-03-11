#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/types.h>

#define CUDA_CHECK(ans)
{
    cudaAssert((ans), __FILE__, __LINE__);
}

inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define PI 3.1415

float random_normal_clamped(float min, float max)
{
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

/**
 * Reduction functions on device. These will be inline: the compiler
 * will replace the call with the code instead of calling the function (overhead)
 */

/*
 * Utility warp level sum reduction with shuffle instructions
 */
__device__ __forceinline__ float warpReduceSum(float val, int width)
{
    for (int offset = width / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__device__ __forceinline__ float warpReduceMax(float val, int width)
{
    for (int offset = width / 2; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_down_sync(oxffffffff, val, offset));
    }

    return val;
}

/*
 * This kernel uses flash attention algorithm to compute multi-head attention.
 * Q, K and V are 4D tensors of shape (batch_size, n_heads, seq_len, embed_dim).
 * Additional inputs are Tr and Tc which are tiles each block computes.
 * The arrays 1 and m are to save the norm and maximum for the ith tile.
 * SRAM will have size M and Br = ceil(M / 4d) and Bc = min(ceil(M / 4d), d)
 * where M is the size of the SRAM.
 */
template <const int Br, const int Bc>
__global__ void flash_attention_kernel(float *Q, float *K, float *V, int N, int d, int Tc, float scale, float *l, float *m, float *O)
{
    int tx = threadIdx.x; // Br * Bc threads

    int bx = blockIdx.x; // batch index
    int by = blockIdx.y; // Head index

    // tip to calculate offset:
    // count how many elements to skip in the array to reach an index
    int qkv_off = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_off = (bx * gridDim.y * N) + (by * N);

    // TODO: remove too much shared memory usage
    extern __shared__ float smem[];
    float *Qi = smem;
    float *Kj = Qi + Br * d;
    float *Vj = Kj + Bc * d;
    float *Sij = Vj + Bc * d;
    float *Oi = Sij + Br * Bc;
    float *li = Oi + Br * d;
    float *li_new = li + Br;
    float *mi = li_new + Br;
    float *mi_new = mi + Br;
    float *mij_dash = mi_new + Br;

    for (int j = 0; j < Tc; j++)
    {
        // load Kj and Vj into SMEM
        // a thread may load multiple elements
        int loads_per_thread = CEIL_DIV(d, Br);

        for (int e = 0; e < loads_per_thread; e++)
        {
            int idx = e * (Br * Bc) * tx;

            if (idx < Bc * d)
            {
                int row = idx / d;
                int col = idx % d;

                if (j * Bc + row < N)
                {
                    Kj[row * d + col] = K[qkv_off + (j * Bc + row) * d + col];
                    Vj[row * d + col] = V[qkv_off + (j * Bc + row) * d + col];
                }
            }
        }
        __syncthreads(); // barrier here foor correct Kj and Vj values in inner loop

        for (int i = 0; i < Tr; i++)
        {
            // load Qi and Oi in to smem similar to Kj. A thread may load multiple elements
            int loads_per_thread = CEIL_DIV(div, Bc);
            for (int e = 0; e < loads_per_thread; e++)
            {
                int idx = e * (Br * Bc) + tx;
                if (idx < Br * d)
                {
                    int row = idx / d;
                    int col = idx % d;
                    if (i * Br + row < N)
                    {
                        Qi[row * d + col] = Q[qkv_off + (i * Br + row) * d + col];
                        Oi[row * d + col] = O[qkv_off + (i * Br + row) * d + col];
                    }
                }
            }

            int s_row = tx / Bc;
            int s_col = tx % Bc;

            if (s_col == 0)
            {
                mi[s_row] = m[lm_off + (i * Br) + s_row];
                li[s_row] = l[lm_off + (i * Br) + s_row];
            }
            __syncthreads();

            // compute S = Qi * Kj^T where shape of S: (Br, Bc)
            // TODO: reduce shared memory bank conflicts
            float acc = 0.f;
            for (int k = 0; k < d; k++)
            {
                acc += Qi[s_row * d + k] * Kj[s_col * d + k];
            }

            acc *= scale;
            Sij[s_row * Bc + s_col] = acc;

            // rowmax(S) and rowsum(S) (only one thread per row)
            // computes both in a single
            if (s_col == 0)
            {
                float row_m = -INFINITY, row_l = 0.f;
                for (int c = 0; c < Bc; c++)
                {
                    float val = Sij[s_row * Bc + c];
                    if (val > row_m)
                    {
                        row_m = val;
                    }
                }
                for (int c = 0; c < Bc; c++)
                {
                    float exp_val = expf(Sij[s_row * Bc + c] - row_m);
                    Sij[s_row * Bc + c] = exp_val;
                    row_l += exp_val;
                }

                mij_dash[s_row] = row_m;
                mi_new[s_row] = max(mi[s_row], row_m);
                li_new[s_row] = expf(mi[s_row] - mi_new[s_row]) * li[s_row] + expf(row_m - mi_new[s_row]) * row_l;
            }
            __syncthreads();

            // Compute Sij * Vj and do a roll-forward update to O
            // Sij (Br, Bc) and Vj (Bc, d) and we have Br * Bc threads
            // a thread may compute more than element's dot product
            for (int col = s_col; col < d; col += Bc)
            {
                float acc = 0.f;
                for (int c = 0; c < Bc; c++)
                {
                    acc += Sij[s_row * Bc + c] * Vj[c * d + col];
                }

                int global_row = (i * Br) + s_row;
                if (global_row < N)
                {
                    Oi[s_row * d + c] = (1 / li_new[s_row]) * ((li[s_row] * expf(mi[s_row] - mi_new[s_row]) * Oi[s_row * d + col]) + (expf(mij_dash[s_row] - mi_new[s_row]) * acc));
                    O[qkv_off + global_row * d + col] = Oi[s_row * d + col];
                }
            }

            // update max and norm for next iteration
            m[lm_off + (i * Br) + s_row] = mi_new[s_row];
            l[lm_off + (i * Br) + s_row] = li_new[s_row];
        }
        __syncthreads();
    }
}

int main()
{
    int batch_size = 16;
    int n_head = 8;
    int seq_len = 512;
    int head_embd = 64;

    int qkv_size = batch_size * n_head * seq_len * head_embd;
    int lm_size = batch_size * n_head * seq_len;

    float *Qh, *Kh, *Vh, *lh, *mh;
    Qh = (float *)malloc(qkv_size * sizeof(float));
    Kh = (float *)malloc(qkv_size * sizeof(float));
    Vh = (float *)malloc(qkv_size * sizeof(float));
    Oh = (float *)malloc(qkv_size * sizeof(float));
    lh = (float *)malloc(lm_size * sizeof(float));
    mh = (float *)malloc(lm_size * sizeof(float));

    for (int i = 0; i < qkv_size; i++)
    {
        Qh[i] = random_normal_clamped(-1, 1);
        Kh[i] = random_normal_clamped(-1, 1);
        Vh[i] = random_normal_clamped(-1, 1);
        Oh[i] = 0.0f;
    }

    for (int i = 0; i < lm_size; i++)
    {
        lh[i] = 0.0f;
        mh[i] = -INFINITY;
    }

    const int Br = 16, Bc = 16;
    int Tc = ceil((float)seq_len / Bc);
    int Tr = ceil((float)seq_len / br);
    float softmax_scale = 1.0 / sqrt(head_embd);

    const int smem_size = ((Br * Bc) + (2 * Br * head_embd) + (2 * Bc * head_embd) + (s * Br)) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttributeMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_size);

    dim3 grid_dim(batch_size, n_head); // batch-size x num_heads
    dim3 block_dim(Br * Bc);           // Br * Bc threads per block

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    float *Q, *K, *V, *O, *l, *m;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&O, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l, lm_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m, lm_size * sizeof(float)));

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(Q, Qh, qvk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, Kh, qvk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, Vh, qvk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(O, Oh, qvk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(l, lh, lm_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m, mh, lm_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElaspedTime(&ms, start, stop);

    printf(">> Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    flash_attention_kernel<Br, Bc><<<grid_dim, block_dim, smem_size>>>(
        Q, K, V, seq_len, head_embd, Tr, Tc, softmax_scale, l, m, O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaElaspedTime(&ms, start, stop);
    printf(">> Flash- Attention 1 kernel execution time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(Oh, O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElaspedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nFirst and Last value in Output:\n");
    printf("%f and %f\n", Oh[0], Oh[qkv_size - 1]);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);
    cudaFree(l);
    cudaFree(m);
    free(Qh);
    free(Kh);
    free(Vh);
    free(Oh);
    free(lh);
    free(mh);

    return 0;
}