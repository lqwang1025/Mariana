/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cuda/alloc_cuda.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-01:15:36:57
 * Description:
 * 
 */

#include <core/alignment.h>
#include <core/utils/logging.h>
#include <core/allocator.h>
#include <core/impl/alloc_cpu.h>
#include <cuda_runtime_api.h>
#include <core/cuda/helper_cuda.h>

namespace mariana {

void* alloc_cuda(size_t nbytes) {
    void *devPtr = nullptr;
    checkCudaErrors(cudaMalloc(&devPtr, nbytes));
    return devPtr;
}

void free_cuda(void* data) {
    if (data) {
        checkCudaErrors(cudaFree(data));
    }
    data = nullptr;
}

} // namespace mariana

