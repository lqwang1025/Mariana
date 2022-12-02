/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/allocator.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:09:24:45
 * Description:
 *
 */

#ifndef __CORE_ALLOCATOR_H__
#define __CORE_ALLOCATOR_H__

#include <mutex>
#include <unordered_map>
#include <core/device.h>

namespace mariana {

class DataPtr {
private:
    void* ptr_;
    Device device_;
public:
    DataPtr() : ptr_(nullptr), device_(DeviceType::CPU) {}
    DataPtr(void* data, Device device) : ptr_(data), device_(device) {}
    void* operator->() const {
        return ptr_;
    }
    operator bool() const {
        return static_cast<bool>(ptr_);
    }
    void* get() const {
        return ptr_;
    }
    Device device() const {
        return device_;
    }
    void unsafe_set_device(Device device) {
        device_ = device;
    }
    void clear() {
        ptr_ = nullptr;
    }
};

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
    return !dp;
}
inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
    return !dp;
}
inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
    return dp;
}
inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
    return dp;
}

class MemoryReporter final {
public:
    MemoryReporter() {}
    void mnew(void* ptr, size_t nbytes);
    void mdelete(void* ptr);
    void outofmemory(size_t nbytes);
private:
    std::mutex mutex_;
    std::unordered_map<void*, size_t> size_table_;
    size_t allocated_ = 0;
};

struct Allocator {
    Allocator()=default;
    virtual ~Allocator()=default;
    virtual DataPtr allocate(size_t n) const = 0;
    virtual void mdelete(DataPtr ptr) const = 0;
};

struct AllocatorContext {
    bool report_memory = false;
    bool fill_zero = false;
    bool fill_junk = false;
};

AllocatorContext& allocator_context();

void set_allocator(DeviceType t, Allocator* alloc, uint8_t priority = 0);

Allocator* get_allocator(const DeviceType& t);

template <DeviceType t>
struct AllocatorRegisterer {
    explicit AllocatorRegisterer(Allocator* alloc) {
        set_allocator(t, alloc);
    }
};

#define REGISTER_ALLOCATOR(t, f)                        \
namespace {                                             \
static AllocatorRegisterer<t> g_allocator_d(f);         \
}

} // namespace mariana

#endif /* __CORE_ALLOCATOR_H__ */

