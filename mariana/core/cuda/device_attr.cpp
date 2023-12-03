/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cuda/device_attr.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-24:09:45:51
 * Description:
 * 
 */

#include <cuda_runtime.h>
#include <core/cuda/device_attr.h>
#include <core/cuda/helper_cuda.h>

namespace mariana {

CudaAttr::CudaAttr() : device_count_(0) {
    checkCudaErrors(cudaGetDeviceCount(&device_count_));
}

int CudaAttr::device_count() const {
    return device_count_;
}

} // namespace mariana

