/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/tensor.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:12:40
 * Description:
 * 
 */

#include <structure/tensor.h>
#include <core/utils/logging.h>
#include <cuda_runtime_api.h>

namespace mariana {

Tensor Tensor::cuda() {
#ifdef WITH_CUDA
    if (this->device() == DeviceType::CUDA) {
        return *this;
    } else if (this->device() == DeviceType::CPU) {
        Tensor tensor(DeviceType::CUDA);
        tensor.set_shape(this->shape().data());
        void* ptr = tensor.impl_->raw_mutable_data(this->dtype());
        cudaMemcpy(ptr, this->data(), this->numel()*this->dtype().itemsize(), cudaMemcpyHostToDevice);
        return tensor;
    } else {
        MLOG(FATAL)<<"Unsupport device to cuda:"<<this->device();
    }
#else
    MLOG(FATAL)<<"Mariana compiling is not with CUDA!";
#endif
}

Tensor Tensor::cpu() {
    if (this->device() == DeviceType::CPU) {
        return *this;
    } else if (this->device() == DeviceType::CUDA) {
#ifdef WITH_CUDA
        Tensor tensor(DeviceType::CPU);
        tensor.set_shape(this->shape().data());
        void* ptr = tensor.impl_->raw_mutable_data(this->dtype());
        cudaMemcpy(ptr, this->data(), this->numel()*this->dtype().itemsize(), cudaMemcpyDeviceToHost);
        return tensor;
#else
        MLOG(FATAL)<<"Mariana compiling is not with CUDA!";
#endif
    } else {
        MLOG(FATAL)<<"Unsupport device to cpu:"<<this->device();
    }
}

// Tensor Tensor::cuda_async(cudaStream_t stream) {
//     if (this->device() == DeviceType::CUDA) {
//         return *this;
//     } else if (this->device() == DeviceType::CPU) {
//         Tensor tensor(DeviceType::CUDA);
//         tensor.set_shape(this->shape().data());
//         void* ptr = tensor.impl_->raw_mutable_data(this->dtype());
//         cudaMemcpyAsync(ptr, this->data(), this->numel()*this->dtype().itemsize(), cudaMemcpyHostToDevice, stream);
//         return tensor;
//     } else {
//         MLOG(FATAL)<<"Unsupport device to cuda:"<<this->device();
//     }
// }

// Tensor Tensor::cpu_async(cudaStream_t stream) {
//     if (this->device() == DeviceType::CPU) {
//         return *this;
//     } else if (this->device() == DeviceType::CUDA) {
//         Tensor tensor(DeviceType::CPU);
//         tensor.set_shape(this->shape().data());
//         void* ptr = tensor.impl_->raw_mutable_data(this->dtype());
//         cudaMemcpyAsync(ptr, this->data(), this->numel()*this->dtype().itemsize(), cudaMemcpyDeviceToHost, stream);
//         return tensor;
//     } else {
//         MLOG(FATAL)<<"Unsupport device to cpu:"<<this->device();
//     }
// }

} // namespace mariana
