/* Copyright (c) Microsoft Corporation. All rights reserved.
   Licensed under the MIT License.

   CUDA implementation of the compute_logpy_post

*/

#include <cuda.h>
#include <iostream>

#include "THC.h"
#include "THCTensor.h"

extern "C" {
    #include "lua.h"
    #include "luaT.h"
    #include "lualib.h"
    #include "lauxlib.h"
};

extern "C" {
    static int compute_logpy_prep(lua_State* L);
    static int compute_logpy_post(lua_State* L);
    int luaopen_libcompute_logpy_lib(lua_State* L);
};

const float loginf = 1000000.0;

template <typename T, typename THCT>
T *getStoragePtr(lua_State* L, THCT * tct)
{
    T *ptr;
    if (tct->storage) {
        ptr = (T*)(tct->storage->data + tct->storageOffset);
    } else {
        lua_pushfstring(L, "THCudaTensor cannot be an empty tensor");
        lua_error(L);
    }
    return ptr;
}

int compute_logpy_prep(lua_State* L)
{
    THCudaTensor *hidden_inputs_tensor = static_cast<THCudaTensor *>(luaT_checkudata(L, 1, "torch.CudaTensor"));
    THCudaTensor *xlength_tensor       = static_cast<THCudaTensor *>(luaT_checkudata(L, 2, "torch.CudaTensor"));
    THCudaTensor *yref_tensor          = static_cast<THCudaTensor *>(luaT_checkudata(L, 3, "torch.CudaTensor"));
    THCudaTensor *ylength_tensor       = static_cast<THCudaTensor *>(luaT_checkudata(L, 4, "torch.CudaTensor"));

    int batch_size = (int)(lua_tonumber(L, 5));
    int T1 = (int)(lua_tonumber(L, 6));
    int T2 = (int)(lua_tonumber(L, 7));

    THCudaTensor *concat_hts_g_tensor    = static_cast<THCudaTensor *>(luaT_checkudata(L, 8, "torch.CudaTensor"));
    THCudaTensor *concat_inputs_g_tensor = static_cast<THCudaTensor *>(luaT_checkudata(L, 9, "torch.CudaTensor"));

    float *hidden_inputs   = getStoragePtr<float, THCudaTensor>(L, hidden_inputs_tensor);
    float *xlength         = getStoragePtr<float, THCudaTensor>(L, xlength_tensor);
    float *yref            = getStoragePtr<float, THCudaTensor>(L, yref_tensor);
    float *ylength         = getStoragePtr<float, THCudaTensor>(L, ylength_tensor);
    float *concat_hts_g    = getStoragePtr<float, THCudaTensor>(L, concat_hts_g_tensor);
    float *concat_inputs_g = getStoragePtr<float, THCudaTensor>(L, concat_inputs_g_tensor);

    

    return 0;
}

__global__ void compute_logpy_post_kernel(  float *t_prob_all, 
                                            float *yref, 
                                            float *ylength, 
                                            float *logpy, 
                                            int *sorted_schedule, 
                                            int s, 
                                            int max_jlen, 
                                            int vocab_size, 
                                            int batch_size,
                                            int batch_max_segment_len,
                                            int T1,
                                            int T2,
                                            int si)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int local_ylength = int(ylength[threadIdx.x]);

    int t       = sorted_schedule[(blockIdx.x + si - 1) * 4];
    int j_start = sorted_schedule[(blockIdx.x + si - 1) * 4 + 1];
    int j_len   = sorted_schedule[(blockIdx.x + si - 1) * 4 + 2];

    float local_t_vec = 0;
    
    if (j_start <= (local_ylength + 1))
    {
        local_t_vec = loginf + t_prob_all[idx * (max_jlen+1) * vocab_size + vocab_size - 1];

        atomicAdd(&(logpy[threadIdx.x * T1 * (T2+1) * (batch_max_segment_len+1) + 
                        (t-1) * (T2+1) * (batch_max_segment_len+1) + 
                        (j_start-1) * (batch_max_segment_len+1) ]), 
                        local_t_vec
                        );
    }

    float tmp_result = 0;
    for (int i = 1; (i < j_len + 1) && (i + j_start <= (local_ylength+1)); i++)
    {
        tmp_result += t_prob_all[idx * (max_jlen+1) * vocab_size + (i-1) * vocab_size + int(yref[threadIdx.x * T2 + j_start + i - 2]) - 1];

        local_t_vec = loginf + tmp_result + 
                      t_prob_all[idx * (max_jlen+1) * vocab_size + (i) * vocab_size + vocab_size - 1];

        atomicAdd(&(logpy[threadIdx.x * T1 * (T2+1) * (batch_max_segment_len+1) + 
                        (t-1) * (T2+1) * (batch_max_segment_len+1) + 
                        (j_start-1) * (batch_max_segment_len+1) + i ]), 
                        local_t_vec
                        );
    }
}

int compute_logpy_post(lua_State* L)
{
    THCudaTensor    *t_prob_all_tensor      = static_cast<THCudaTensor *>(luaT_checkudata(L, 1, "torch.CudaTensor"));
    THCudaTensor    *yref_tensor            = static_cast<THCudaTensor *>(luaT_checkudata(L, 2, "torch.CudaTensor"));
    THCudaTensor    *ylength_tensor         = static_cast<THCudaTensor *>(luaT_checkudata(L, 3, "torch.CudaTensor"));
    THCudaTensor    *logpy_tensor           = static_cast<THCudaTensor *>(luaT_checkudata(L, 4, "torch.CudaTensor"));
    THCudaIntTensor *sorted_schedule_tensor = static_cast<THCudaIntTensor *>(luaT_checkudata(L, 5, "torch.CudaIntTensor"));

    int s          = (int)(lua_tonumber(L, 6));
    int max_jlen   = (int)(lua_tonumber(L, 7));
    int vocab_size = (int)(lua_tonumber(L, 8));
    int batch_size = (int)(lua_tonumber(L, 9));
    int batch_max_segment_len = (int)(lua_tonumber(L, 10));
    int T1 = (int)(lua_tonumber(L, 11));
    int T2 = (int)(lua_tonumber(L, 12));
    int si = (int)(lua_tonumber(L, 13));

    float *t_prob_all    = getStoragePtr<float, THCudaTensor>(L, t_prob_all_tensor);
    float *yref          = getStoragePtr<float, THCudaTensor>(L, yref_tensor);
    float *ylength       = getStoragePtr<float, THCudaTensor>(L, ylength_tensor);
    float *logpy         = getStoragePtr<float, THCudaTensor>(L, logpy_tensor);
    int *sorted_schedule = getStoragePtr<int, THCudaIntTensor>(L, sorted_schedule_tensor);
    
    dim3 blockDim(batch_size);
    dim3 gridDim(s);
    compute_logpy_post_kernel<<<gridDim, blockDim>>>(t_prob_all, yref, ylength, logpy, sorted_schedule, 
                                                    s, max_jlen, vocab_size, batch_size, batch_max_segment_len,
                                                    T1, T2, si);

    cudaDeviceSynchronize();

    return 0;
}

int luaopen_libcompute_logpy_lib(lua_State* L) {
    lua_register(L, "compute_logpy_prep", compute_logpy_prep);
    lua_register(L, "compute_logpy_post", compute_logpy_post);
    return 0;
}

