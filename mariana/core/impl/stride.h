/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/stride.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-12:16:40:51
 * Description:
 *
 */

#ifndef __CORE_IMPL_STRIDE_H__
#define __CORE_IMPL_STRIDE_H__

#include <core/impl/shape.h>

namespace mariana {

class Stride final {
public:
    Stride() : dims_(0) {
        memset(stride_, 0, sizeof(int64_t)*MAX_DIMS);
    }
    ~Stride()=default;
    Stride(IntArrayRef stride) {
        set(stride);
    }
    Stride(ArrayRef<int32_t> stride) {
        set(stride);
    }
    Stride(const std::initializer_list<int64_t>& Vec) {
        if (std::begin(Vec) == std::end(Vec)) {
            return;
        }
        IntArrayRef stride{Vec};
        set(stride);
    }
    Stride& operator=(IntArrayRef stride) {
        return set(stride);
    }
    Stride& operator=(ArrayRef<int32_t> stride) {
        return set(stride);
    }
    Stride& operator=(const Stride& rhs) {
        if (this == &rhs) return *this;
        IntArrayRef stride{rhs.stride_, rhs.dims_};
        return set(stride);
    }
    IntArrayRef data() const {
        return IntArrayRef(stride_, dims_);
    }
    uint8_t dims() const {
        return dims_;
    }
    bool empty() const {
        return dims_ == 0;
    }
    bool equals(const Stride& rhs) const {
        return make_arrayref(stride_, dims_)==make_arrayref(rhs.stride_, rhs.dims_);
    }
    int64_t operator[](uint8_t idx) const {
        MCHECK(idx<dims_)<<"Stride out of range: idx:"<<(int)idx
                         <<" dims_:"<<dims_;
        return stride_[idx];
    }
    int64_t& operator[](uint8_t idx) {
        MCHECK(idx<dims_)<<"Stride out of range: idx:"<<(int)idx
                         <<" dims_:"<<dims_;
        return stride_[idx];
    }
    Stride& set(IntArrayRef stride);
    Stride& set(ArrayRef<int32_t> stride);
private:
    int64_t stride_[MAX_DIMS];
    uint8_t dims_;
};

bool operator==(const Stride& a, const Stride& b);

bool operator==(const Stride& a, IntArrayRef b);

bool operator==(const Stride& a, ArrayRef<int32_t> b);

bool operator!=(const Stride& a, const Stride& b);

bool operator!=(const Stride& a, IntArrayRef b);

bool operator!=(const Stride& a, ArrayRef<int32_t> b);

} // namespace mariana

#endif /* __CORE_IMPL_STRIDE_H__ */

