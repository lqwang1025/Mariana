/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_impl.h
 * Authors    : lqwang@pandora
 * Create Time: 2022-11-29:20:49:13
 * Description:
 *
 */

#ifndef __CORE_TENSOR_IMPL_H__
#define __CORE_TENSOR_IMPL_H__

#include <string>
#include <core/storage.h>
#include <core/impl/shape.h>
#include <core/impl/stride.h>
#include <core/utils/logging.h>
#include <core/utils/typemeta.h>

namespace mariana {

class TensorImpl {
public:
    TensorImpl(Storage&& storage, const TypeMeta data_type) {
        data_type_ = data_type;
        storage_ = storage;
        device_ = storage.device();
    }
    TensorImpl(Device device=Device{DeviceType::CPU}) : device_(device) {}
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;
    virtual ~TensorImpl(){}
    void reshape(IntArrayRef shape);
    void share_data(const TensorImpl& src);
    void ShareExternalPointer(DataPtr&& data_ptr, TypeMeta data_type, size_t size_bytes);
    void set_shape_and_stride(IntArrayRef shape, IntArrayRef stride,
                              int64_t storage_offset = 0 ) {
        MCHECK(shape.size()==stride.size())<<"dimensionality of shape ("
                                            <<shape.size()
                                            <<") must match dimensionality of strides ("
                                            <<stride.size()<<")";
        shape_.set(shape);
        stride_.set(stride);
        storage_offset_ = storage_offset;
    }
    void set_shape(IntArrayRef shape, int64_t storage_offset = 0) {
        const auto new_dim = shape.size();
        shape_.set(shape);
        int64_t _stride[new_dim];
        for (size_t dim = 0; dim < new_dim; ++dim) {
            int64_t stride = shape_.stride_at(dim);
            _stride[dim] = stride;
        }
        stride_.set({_stride, new_dim});
        storage_offset_ = storage_offset;
    }
    virtual void set_storage_offset(int64_t storage_offset) {
        storage_offset_ = storage_offset;
    }
    void set_name(const std::string& name) {
        name_ = name;
    }
    const std::string& name() const {
        return name_;
    }
    Shape shape() const {
        return shape_;
    }
    Stride stride() const {
        return stride_;
    }
    int64_t shape(uint8_t idx) const {
        return shape_[idx];
    }
    int64_t stride(uint8_t idx) const {
        return stride_[idx];
    }
    int64_t numel() const {
        return shape_.size();
    }
    size_t dim() const {
        return shape_.dims();
    }
    int64_t storage_offset() const {
        return storage_offset_;
    }
    const Storage& storage() const {
        return storage_;
    }
    Device device() const {
        return device_;
    }
    bool storage_initialized() const {
        return storage_.initialized();
    }
    bool dtype_initialized() const noexcept {
        return data_type_ != TypeMeta();
    }
    inline bool is_empty() const {
        return numel() == 0;
    }
    template <typename T>
    inline T* data_ptr_impl() const {
        MCHECK(storage_initialized())<<"Tensor storage is not allocated yet.";
        return storage_.unsafe_data<T>()+storage_offset_;
    }
    inline void* data() const {
        MCHECK(storage_initialized())<<"Tensor storage is not allocated yet.";
        if (is_empty()) {
            return nullptr;
        }
        return static_cast<void*>(static_cast<char*>(storage_.data()) +
                                  data_type_.itemsize() * storage_offset_);
    }
    template <typename T>
    inline T* unsafe_data() const {
        return storage_.unsafe_data<T>() + storage_offset_;
    }
    const TypeMeta dtype() const {
        return data_type_;
    }
    size_t itemsize() const {
        MCHECK(dtype_initialized())<<"Cannot report itemsize of "
            "Tensor that doesn't have initialized dtype ";
        return data_type_.itemsize();
    }
    inline void* raw_mutable_data(TypeMeta meta) {
        if (data_type_ == meta && storage_initialized()) {
            return static_cast<void*>(static_cast<char*>(storage_.data()) +
                                      storage_offset_ * meta.itemsize());
        } else {
            storage_offset_ = 0;
            data_type_ = meta;
            Allocator* alloc = get_allocator(device_.type());
            auto byte_size = numel()*data_type_.itemsize();
            storage_ = Storage{byte_size, alloc};
            return storage_.data();
        }
    }
    template <typename T>
    inline T* mutable_data() {
        if (storage_initialized() && data_type_.match<T>()) {
            return static_cast<T*>(storage_.data()) + storage_offset_;
        }
        return static_cast<T*>(raw_mutable_data(TypeMeta::make<T>()));
    }
private:
    Shape shape_;
    Stride stride_;
    TypeMeta data_type_;
    Storage storage_;
    Device device_;
    std::string name_ = "";
    int64_t storage_offset_ = 0;
};

} // namespace mariana
#endif /* __CORE_TENSOR_IMPL_H__ */

