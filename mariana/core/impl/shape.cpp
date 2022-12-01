/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/shape.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:15:25:44
 * Description:
 * 
 */

#include <core/impl/shape.h>
#include <core/utils/logging.h>
#include <core/macros/macros.h>

namespace mariana {

Shape& Shape::set(ArrayRef<int32_t> shape) {
    int64_t data[shape.size()];
    for (size_t i = 0; i < shape.size(); ++i) {
        data[i] = shape[i];
    }
    return set(IntArrayRef(data, shape.size()));
}

Shape& Shape::set(IntArrayRef shape) {
    MCHECK(shape.size() <= MAX_DIMS)<<"Shape max dims is "<<MAX_DIMS
                                    <<" your shape dims is"<<shape.size();
    size_ = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (MAR_UNLIKELY(shape[i] < 0)) {
            MLOG(FATAL)<<"Shape do not support negtive value:"<<i<<" "<<shape[i];
        }
        shape_[i] = shape[i];
        size_ *= shape_[i];
    }
    dims_ = shape.size();
    return *this;
}


bool operator==(const Shape& a, const Shape& b) {
    return a.equals(b);
}

bool operator==(const Shape& a, IntArrayRef b) {
    return a.equals(Shape(b));
}

bool operator==(const Shape& a, ArrayRef<int32_t> b) {
    int64_t data[b.size()];
    for (size_t i = 0; i < b.size(); ++i) {
        data[i] = b[i];
    }
    return a.equals(Shape(IntArrayRef{data, b.size()}));
}

bool operator!=(const Shape& a, const Shape& b) {
    return !a.equals(b);
}

bool operator!=(const Shape& a, IntArrayRef b) {
    return !a.equals(Shape(b));
}

bool operator!=(const Shape& a, ArrayRef<int32_t> b) {
    int64_t data[b.size()];
    for (size_t i = 0; i < b.size(); ++i) {
        data[i] = b[i];
    }
    return !a.equals(Shape(IntArrayRef{data, b.size()}));
}

} // namespace mariana
