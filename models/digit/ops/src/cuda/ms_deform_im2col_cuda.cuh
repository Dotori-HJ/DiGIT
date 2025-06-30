/*!
**************************************************************************
* Modified from Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_linear(
    const scalar_t* &bottom_data, 
    const int &width, const int &nheads, const int &channels,
    const scalar_t &w, const int &m, const int &c
)
{
  const int w_low = floor(w);
  const int w_high = w_low + 1;

  const scalar_t lw = w - w_low;
  const scalar_t hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (w_low >= 0)
  {
    const int ptr1 = w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (w_high <= width - 1)
  {
    const int ptr2 = w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }

  const scalar_t w1 = hw, w2 = lw;
  const scalar_t val = (w1 * v1 + w2 * v2);
  return val;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_linear(
    const scalar_t* &bottom_data, 
    const int &width, const int &nheads, const int &channels,
    const scalar_t &w, const int &m, const int &c,
    const scalar_t &top_grad,
    const scalar_t &attn_weight,
    scalar_t* &grad_value, 
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight
)
{
  const int w_low = floor(w);
  const int w_high = w_low + 1;

  const scalar_t lw = w - w_low;
  const scalar_t hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hw, w2 = lw;
  const scalar_t top_grad_value = top_grad * attn_weight;

  scalar_t grad_weight = 0;
  scalar_t v1 = 0;
  if (w_low >= 0)
  {
    const int ptr1 = w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_weight -= v1;
    atomicAdd(grad_value+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (w_high <= width - 1)
  {
    const int ptr2 = w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_weight += v2;
    atomicAdd(grad_value+ptr2, w2*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_weight * top_grad_value;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_linear_gm(
    const scalar_t* &bottom_data, 
    const int &width, const int &nheads, const int &channels,
    const scalar_t &w, const int &m, const int &c,
    const scalar_t &top_grad,
    const scalar_t &attn_weight,
    scalar_t* &grad_value, 
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight
)
{
  const int w_low = floor(w);
  const int w_high = w_low + 1;

  const scalar_t lw = w - w_low;
  const scalar_t hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hw, w2 = lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_weight = 0;

  scalar_t v1 = 0;
  if (w_low >= 0)
  {
    const int ptr1 = w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_weight -= v1;
    atomicAdd(grad_value+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (w_high <= width - 1)
  {
    const int ptr2 = w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_weight += v2;
    atomicAdd(grad_value+ptr2, w2*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2);
  atomicAdd(grad_attn_weight, top_grad * val); 
  atomicAdd(grad_sampling_loc, width * grad_weight * top_grad_value);
}


template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t *data_value, 
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, 
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *data_col
)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col;
      const int spatial_w = data_spatial_shapes[spatial_ptr];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (w_im > -1 && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_linear(data_value_ptr, spatial_w, num_heads, channels, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 1;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n,
    const scalar_t *grad_col,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, 
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_value,
    scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight
)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col;
      const int spatial_w = data_spatial_shapes[spatial_ptr];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+threadIdx.x) = 0;
        *(cache_grad_attn_weight+threadIdx.x) = 0;
        if (w_im > -1 && w_im < spatial_w)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_w, num_heads, channels, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_a=cache_grad_attn_weight[0];
          int sid=1;
          for (unsigned int tid = 1; tid < blockSize; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[tid];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 1;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n,
    const scalar_t *grad_col,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, 
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_value,
    scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight
)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col;
      const int spatial_w = data_spatial_shapes[spatial_ptr];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+threadIdx.x) = 0;
        *(cache_grad_attn_weight+threadIdx.x) = 0;
        if (w_im > -1 && w_im < spatial_w)
        {
          ms_deform_attn_col2im_linear_gm(
            data_value_ptr, spatial_w, num_heads, channels, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockSize/2; s>0; s>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid;
            const unsigned int xid2 = tid + s;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
          }
          __syncthreads();
        }

        if (tid == 0)
        { 
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n,
    const scalar_t *grad_col,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, 
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_value,
    scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight
)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col;
      const int spatial_w = data_spatial_shapes[spatial_ptr];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+threadIdx.x) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (w_im > -1 && w_im < spatial_w)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_w, num_heads, channels, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_a=cache_grad_attn_weight[0];
          int sid=1;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[tid];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 1;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(
    const int n,
    const scalar_t *grad_col,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, 
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_value,
    scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight
)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col;
      const int spatial_w = data_spatial_shapes[spatial_ptr];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (w_im > -1 && w_im < spatial_w)
        {
          ms_deform_attn_col2im_linear_gm(
            data_value_ptr, spatial_w, num_heads, channels, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes, 
    const int64_t* data_level_start_index, 
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size, 
    const int num_heads, 
    const int channels, 
    const int num_levels, 
    const int num_query,
    const int num_point,
    scalar_t* data_col
)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col
  );
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ms_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t * data_spatial_shapes,
    const int64_t * data_level_start_index,
    const scalar_t * data_sampling_loc,
    const scalar_t * data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels,
    const int num_query,
    const int num_point, 
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight
)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;

  switch(channels)
  {
        case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        break;
        default:
        printf("Error: Unsupported block size %d\n", num_threads);
        assert(false && "Unsupported block size");
    }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}