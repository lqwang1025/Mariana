/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_options.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:36:22
 * Description:
 *
 */

#ifndef __CORE_TENSOR_OPTIONS_H__
#define __CORE_TENSOR_OPTIONS_H__

#include <core/device.h>
#include <core/layout.h>
#include <core/utils/typemeta.h>

namespace mariana {

struct TensorOption {
    TensorOption() : has_device_(false),
                     has_dtype_(false),
                     has_layout_(false) {}
    TensorOption(Layout layout) : TensorOption() {
        this->set_layout(layout);
    }
    TensorOption(TypeMeta dtype) : TensorOption() {
        this->set_dtype(dtype);
    }
    TensorOption(Device device) : TensorOption() {
        this->set_device(device);
    }
    Device device() const {
        return device_;
    }
    Layout layout() const {
        return layout_;
    }
    TypeMeta dtype() const {
        return dtype_;
    }
    bool has_layout() const {
        return has_layout_;
    }
    bool has_device() const {
        return has_device_;
    }
    bool has_dtype() const {
        return has_dtype_;
    }
private:
    void set_device(Device device) {
        device_ = device;
        has_device_ = true;
    }
    void set_layout(Layout layout) {
        layout_ = layout;
        has_layout_ = true;
    }
    void set_dtype(TypeMeta dtype) {
        dtype_ = dtype;
        has_dtype_ = true;
    }
private:
    Device device_ = kCPU; // 8-bit
    TypeMeta dtype_=TypeMeta::make<float>(); // 16-bit
    Layout layout_ = kStrided; // 8-bit
    bool has_device_ : 1;
    bool has_dtype_ : 1;
    bool has_layout_ : 1;
};

} // namespace mariana

#endif /* __CORE_TENSOR_OPTIONS_H__ */

