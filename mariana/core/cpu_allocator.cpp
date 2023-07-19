/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : cpu_allocator.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:14:01:48
 * Description:
 * 
 */

#include <core/macros/macros.h>
#include <core/allocator.h>
#include <core/cpu_allocator.h>
#include <core/impl/alloc_cpu.h>
#include <core/utils/logging.h>

namespace mariana {

struct CpuAllocator final : public Allocator {
    CpuAllocator() {}
    
    DataPtr allocate(size_t n) const override {
        void* data = nullptr;
        data = alloc_cpu(n);
        if (data == nullptr) {
            cpu_memory_reporter().outofmemory(n);
            CHECK(false)<<"Attempt request:"<<n<< " out of memory";
        }
        cpu_memory_reporter().mnew(data, n);
        return {data, Device(DeviceType::CPU)};
    }
    
    void mdelete(DataPtr ptr) const override {
        if (ptr.get() == nullptr) return;
        if (MAR_UNLIKELY(ptr.device()!=DeviceType::CPU)) return;
        cpu_memory_reporter().mdelete(ptr.get());
        free_cpu(ptr.get());
    }
};

MemoryReporter& cpu_memory_reporter() {
    static MemoryReporter cpu_reporter_;
    return cpu_reporter_;
}

Allocator* get_cpu_allocator() {
    return get_allocator(DeviceType::CPU);
}

void set_cpu_allocator(Allocator* alloc, uint8_t priority) {
    set_allocator(DeviceType::CPU, alloc, priority);
}

static CpuAllocator g_cpu_alloc;

REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_alloc);

} // namespace mariana
