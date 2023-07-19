/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/storage_impl.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:16:16:07
 * Description:
 *
 */

#ifndef __CORE_STORAGE_IMPL_H__
#define __CORE_STORAGE_IMPL_H__

#include <atomic>
#include <core/allocator.h>

namespace mariana {

struct StorageImpl {
    StorageImpl(size_t size_bytes, DataPtr data_ptr, Allocator* allocator) :
        size_bytes_(size_bytes), data_ptr_(std::move(data_ptr)),
        allocator_(allocator), own_data_(false) {}
    StorageImpl(size_t size_bytes, Allocator* allocator) :
        StorageImpl(size_bytes, allocator->allocate(size_bytes), allocator) {
        own_data_ = true;
    }
    StorageImpl& operator=(StorageImpl&& other) = default;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl() = delete;
    StorageImpl(StorageImpl&& other) = default;
    StorageImpl(const StorageImpl&) = delete;
    ~StorageImpl() {
        if (own_data_ == true) {
            allocator_->mdelete(data_ptr_);
        }
    }
    template <typename T>
    inline T* data() const {
        return unsafe_data<T>();
    }
    template <typename T>
    inline T* unsafe_data() const {
        return static_cast<T*>(this->data_ptr_.get());
    }
    size_t nbytes() const {
        return size_bytes_;
    }
    void set_nbytes(size_t size_bytes) {
        size_bytes_ = size_bytes;
    }
    DataPtr& data_ptr() {
        return data_ptr_;
    }
    DataPtr set_data_ptr(DataPtr&& data_ptr) {
        DataPtr old_data_ptr(std::move(data_ptr_));
        data_ptr_ = std::move(data_ptr);
        return old_data_ptr;
    }
    const DataPtr& data_ptr() const {
        return data_ptr_;
    }
    void* data() {
        return data_ptr_.get();
    }
    void* data() const {
        return data_ptr_.get();
    }
    DeviceType device_type() const {
        return data_ptr_.device().type();
    }
    Device device() const {
        return data_ptr_.device();
    }
    Allocator* allocator() {
        return allocator_;
    }
    const Allocator* allocator() const {
        return allocator_;
    }
    void set_allocator(Allocator* allocator) {
        allocator_ = allocator;
    }
private:
    size_t size_bytes_;
    DataPtr data_ptr_;
    Allocator* allocator_;
    std::atomic<bool> own_data_;
};

} // namespace mariana

#endif /* __CORE_STORAGE_IMPL_H__ */

