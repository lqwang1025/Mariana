/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/shape.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:15:10:31
 * Description:
 *
 */

#ifndef __CORE_IMPL_SHAPE_H__
#define __CORE_IMPL_SHAPE_H__

#include <core/utils/arrary_ref.h>

namespace mariana {

constexpr size_t MAX_DIMS = 8;

class Shape final {
public:
    Shape() : dims_(0), size_(0) {
        memset(shape_, 0, sizeof(int64_t)*MAX_DIMS);
    }
    ~Shape()=default;
    Shape(IntArrayRef shape) {
        set(shape);
    }
    Shape(ArrayRef<int32_t> shape) {
        set(shape);
    }
    Shape(const std::initializer_list<int64_t>& Vec) {
        if (std::begin(Vec) == std::end(Vec)) {
            return;
        }
        IntArrayRef shape{Vec};
        set(shape);
    }
    Shape& set(IntArrayRef shape);
    Shape& set(ArrayRef<int32_t> shape);
    Shape& reshape(IntArrayRef shape) {
        return set(shape);
    }
    Shape& reshape(ArrayRef<int32_t> shape) {
        return set(shape);
    }
    Shape& operator=(IntArrayRef shape) {
        return set(shape);
    }
    Shape& operator=(ArrayRef<int32_t> shape) {
        return set(shape);
    }
    Shape& operator=(const Shape& rhs) {
        if (this == &rhs) return *this;
        IntArrayRef shape{rhs.shape_, rhs.dims_};
        return set(shape);
    }
    IntArrayRef data() const {
        return IntArrayRef(shape_, dims_);
    }
    size_t dims() const {
        return dims_;
    }
    bool empty() const {
        return dims_ == 0;
    }
    int64_t size() const {
        return size_;
    }
    bool equals(const Shape& rhs) const {
        return make_arrayref(shape_, dims_)==make_arrayref(rhs.shape_, rhs.dims_);
    }
    int64_t operator[](uint8_t idx) const {
        MCHECK(idx<dims_)<<"Shape out of range: idx:"<<(int)idx
                         <<" dims_:"<<dims_;
        return shape_[idx];
    }
    int64_t stride_at(uint8_t idx) const {
        MCHECK(idx<dims_)<<"Shape out of range: idx:"<<(int)idx
                         <<" dims_:"<<dims_;
        IntArrayRef __a = data();
        int64_t stride = 1;
        for (auto it = __a.end()-1; it != __a.begin()+idx; --it) {
            stride *= *it;
        }
        return stride;
    }
    
private:
    int64_t shape_[MAX_DIMS];
    size_t dims_;
    int64_t size_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);

bool operator==(const Shape& a, const Shape& b);

bool operator==(const Shape& a, IntArrayRef b);

bool operator==(const Shape& a, ArrayRef<int32_t> b);

bool operator!=(const Shape& a, const Shape& b);

bool operator!=(const Shape& a, IntArrayRef b);

bool operator!=(const Shape& a, ArrayRef<int32_t> b);

} // namespace mariana

#endif /* __CORE_IMPL_SHAPE_H__ */

