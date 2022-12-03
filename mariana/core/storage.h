/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/storage.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:16:52:37
 * Description:
 *
 */

#ifndef __CORE_STORAGE_H__
#define __CORE_STORAGE_H__

#include <memory>
#include <core/storage_impl.h>

namespace mariana {

struct Storage {
    Storage() : storage_impl_(nullptr) {}
    Storage(std::shared_ptr<StorageImpl> ptr) : storage_impl_(std::move(ptr)) {}
    Storage(size_t size_bytes, Allocator* allocator)
        : storage_impl_(new StorageImpl{size_bytes, allocator}) {}
    Storage(size_t size_bytes, DataPtr data_ptr, Allocator* allocator) :
        storage_impl_(new StorageImpl{size_bytes, data_ptr, allocator}) {}
    ~Storage() {}
    template <typename T>
    T* data() const {
        return storage_impl_->data<T>();
    }
    template <typename T>
    T* unsafe_data() const {
        return storage_impl_->unsafe_data<T>();
    }
    void set_nbytes(size_t size_bytes) const {
        storage_impl_->set_nbytes(size_bytes);
    }
    size_t nbytes() const {
        return storage_impl_->nbytes();
    }
    void* data() const {
        return storage_impl_->data();
    }

    DataPtr& data_ptr() {
        return storage_impl_->data_ptr();
    }

    const DataPtr& data_ptr() const {
        return storage_impl_->data_ptr();
    }
    DataPtr set_data_ptr(DataPtr&& data_ptr) const {
        return storage_impl_->set_data_ptr(std::move(data_ptr));
    }
    DeviceType device_type() const {
        return storage_impl_->device_type();
    }
    Allocator* allocator() const {
        return storage_impl_->allocator();
    }
    Device device() const {
        return storage_impl_->device();
    }
    std::shared_ptr<StorageImpl> unsafe_get_storageimpl() const {
        return storage_impl_;
    }
    bool is_alias_of(const Storage& other) const {
        return storage_impl_ == other.storage_impl_;
    }
    size_t use_count() const {
        return storage_impl_.use_count();
    }
    Storage& operator= (const Storage& rhs) {
        storage_impl_ = rhs.storage_impl_;
        return *this;
    }
private:
    std::shared_ptr<StorageImpl> storage_impl_;
};

} // namespace mariana

#endif /* __CORE_STORAGE_H__ */

