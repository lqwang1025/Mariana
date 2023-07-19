/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/allocator.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:10:18:51
 * Description:
 * 
 */

#include <core/utils/logging.h>
#include <core/device_type.h>
#include <core/allocator.h>

namespace mariana {

Allocator* allocator_array[COMPILE_TIME_MAX_DEVICE_TYPES] = {nullptr};
uint8_t allocator_priority[COMPILE_TIME_MAX_DEVICE_TYPES] = {0};

AllocatorContext& allocator_context() {
    static AllocatorContext allocator_context_;
    return allocator_context_;
}

void set_allocator(DeviceType t, Allocator* alloc, uint8_t priority) {
    if (priority >= allocator_priority[static_cast<int>(t)]) {
        allocator_array[static_cast<int>(t)] = alloc;
        allocator_priority[static_cast<int>(t)] = priority;
    }
}

Allocator* get_allocator(const DeviceType& t) {
    Allocator* alloc = allocator_array[static_cast<int>(t)];
    MCHECK_NOTNULL(alloc);
    return alloc;
}

void MemoryReporter::mnew(void* ptr, size_t nbytes) {
    if (nbytes == 0) return;
    size_t allocated = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        size_table_[ptr] = nbytes;
        allocated_ += nbytes;
        allocated = allocated_;
    }
    if (allocator_context().report_memory) {
        MLOG(INFO) << "mariana alloc " << nbytes << " bytes, total alloc " << allocated
                  << " bytes.";
    }
}

void MemoryReporter::mdelete(void* ptr) {
    size_t nbytes = 0;
    size_t allocated = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = size_table_.find(ptr);
        if (it != size_table_.end()) {
            allocated_ -= it->second;
            allocated = allocated_;
            nbytes = it->second;
            size_table_.erase(it);
        }
    }
    if (allocator_context().report_memory) {
        MLOG(INFO) << "Mariana deleted " << nbytes << " bytes, total alloc " << allocated
                   << " bytes.";
    }
}

void MemoryReporter::outofmemory(size_t nbytes) {
    if (nbytes == 0) return;
    size_t allocated = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        allocated = allocated_;
    }
    if (allocator_context().report_memory) {
        MLOG(INFO) << "Mariana Out of Memory. Trying to allocate " << nbytes
                   << " bytes, total alloc " << allocated << " bytes.";
    }
}

} // namespace mariana
