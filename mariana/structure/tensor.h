/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/tensor.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:11:20
 * Description:
 *
 */

#ifndef __STRUCTURE_TENSOR_H__
#define __STRUCTURE_TENSOR_H__

#include <core/tensor_impl.h>
#include <core/utils/logging.h>

namespace mariana {

class Tensor {
public:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
        if (impl_.get() == nullptr) {
            //TODO:ERROR
        }
    }
    Tensor(Device device=Device{DeviceType::CPU}) {
        impl_ = std::make_shared<TensorImpl>(device);
    }
    Tensor(const Tensor&)=default;
    Tensor(Tensor&&)=default;
    ~Tensor()=default;
    void set_shape(IntArrayRef shape, int64_t storage_offset = 0) {
        impl_->set_shape(shape, storage_offset);
    }
    void set_shape_and_stride(IntArrayRef shape, IntArrayRef stride,
                              int64_t storage_offset = 0 ) {
        impl_->set_shape_and_stride(shape, stride, storage_offset);
    }
    void set_storage_offset(int64_t storage_offset) {
        impl_->set_storage_offset(storage_offset);
    }

    Shape shape() const {
        return impl_->shape();
    }
    Stride stride() const {
        return impl_->stride();
    }
    int64_t shape(uint8_t idx) const {
        return impl_->shape(idx);
    }
    int64_t stride(uint8_t idx) const {
        return impl_->stride(idx);
    }
    int64_t numel() const {
        return impl_->numel();
    }
    size_t dim() const {
        return impl_->dim();
    }
    int64_t storage_offset() const {
        return impl_->storage_offset();
    }
    const Storage& storage() const {
        return impl_->storage();
    }
    Device device() const {
        return impl_->device();
    }
    inline bool is_empty() const {
        return impl_->is_empty();
    }
    template <typename T>
    inline T* data_ptr_impl() const {
        return impl_->data_ptr_impl<T>();
    }
    inline void* data() const {
        return impl_->data();
    }
    template <typename T>
    inline T* unsafe_data() const {
        return impl_->unsafe_data<T>();
    }
    const TypeMeta dtype() const {
        return impl_->dtype();
    }
    size_t itemsize() const {
        return impl_->itemsize();
    }
    template <typename T>
    inline T* mutable_data() {
        return impl_->mutable_data<T>();
    }
    inline size_t use_count() const {
        return impl_.use_count();
    }
    Tensor& operator=(const Tensor& rhs) {
        if (this == &rhs) return *this;
        impl_ = rhs.impl_;
        return *this;
    }
    Tensor cuda();
    Tensor cpu();
private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace mariana

#endif /* __STRUCTURE_TENSOR_H__ */

