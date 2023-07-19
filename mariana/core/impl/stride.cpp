/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/stride.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-12:16:47:39
 * Description:
 * 
 */

#include <core/impl/stride.h>
#include <core/utils/logging.h>
#include <core/macros/macros.h>

namespace mariana {

Stride& Stride::set(ArrayRef<int32_t> stride) {
    int64_t data[stride.size()];
    for (size_t i = 0; i < stride.size(); ++i) {
        data[i] = stride[i];
    }
    return set(IntArrayRef(data, stride.size()));
}

Stride& Stride::set(IntArrayRef stride) {
    MCHECK(stride.size() <= MAX_DIMS)<<"Stride max dims is "<<MAX_DIMS
                                    <<" your stride dims is"<<stride.size();
    for (size_t i = 0; i < stride.size(); ++i) {
        if (MAR_UNLIKELY(stride[i] < 0)) {
            MLOG(FATAL)<<"Stride do not support negtive value:"<<i<<" "<<stride[i];
        }
        stride_[i] = stride[i];
    }
    dims_ = stride.size();
    return *this;
}

bool operator==(const Stride& a, const Stride& b) {
    return a.equals(b);
}

bool operator==(const Stride& a, IntArrayRef b) {
    return a.equals(Stride(b));
}

bool operator==(const Stride& a, ArrayRef<int32_t> b) {
    int64_t data[b.size()];
    for (size_t i = 0; i < b.size(); ++i) {
        data[i] = b[i];
    }
    return a.equals(Stride(IntArrayRef{data, b.size()}));
}

bool operator!=(const Stride& a, const Stride& b) {
    return !a.equals(b);
}

bool operator!=(const Stride& a, IntArrayRef b) {
    return !a.equals(Stride(b));
}

bool operator!=(const Stride& a, ArrayRef<int32_t> b) {
    int64_t data[b.size()];
    for (size_t i = 0; i < b.size(); ++i) {
        data[i] = b[i];
    }
    return !a.equals(Stride(IntArrayRef{data, b.size()}));
}

std::ostream& operator<<(std::ostream& out, const Stride& stride) {
    out<<stride.data();
    return out;
}

} // namespace mariana
