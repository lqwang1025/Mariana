/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cuda/cuda_allocator.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-01:15:55:44
 * Description:
 * 
 */

#include <core/macros/macros.h>
#include <core/cuda/cuda_allocator.h>
#include <core/cuda/alloc_cuda.h>
#include <core/utils/logging.h>

namespace mariana {

struct CudaAllocator final : public Allocator {
    CudaAllocator() {}
    
    DataPtr allocate(size_t n) const override {
        void* data = nullptr;
        data = alloc_cuda(n);
        if (data == nullptr) {
            cuda_memory_reporter().outofmemory(n);
            CHECK(false)<<"Attempt request:"<<n<< " out of memory";
        }
        cuda_memory_reporter().mnew(data, n);
        return {data, Device(DeviceType::CUDA)};
    }
    
    void mdelete(DataPtr ptr) const override {
        if (ptr.get() == nullptr) return;
        if (MAR_UNLIKELY(ptr.device()!=DeviceType::CUDA)) return;
        cuda_memory_reporter().mdelete(ptr.get());
        free_cuda(ptr.get());
    }
};

MemoryReporter& cuda_memory_reporter() {
    static MemoryReporter cuda_reporter_;
    return cuda_reporter_;
}

Allocator* get_cuda_allocator() {
    return get_allocator(DeviceType::CUDA);
}

void set_cuda_allocator(Allocator* alloc, uint8_t priority) {
    set_allocator(DeviceType::CUDA, alloc, priority);
}

static CudaAllocator g_cuda_alloc;

REGISTER_ALLOCATOR(DeviceType::CUDA, &g_cuda_alloc);


} // namespace mariana
