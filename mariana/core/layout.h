/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/layout.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:54:33
 * Description:
 *
 */

#ifndef __CORE_LAYOUT_H__
#define __CORE_LAYOUT_H__

#include <ostream>
#include <core/utils/logging.h>

namespace mariana {

enum class Layout : int8_t {
    Strided,
    Sparse,
    SparseCsr,
    SparseCsc,
    SparseBsr,
    SparseBsc,
    NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;

inline std::ostream& operator<<(std::ostream& stream, Layout layout) {
    switch (layout) {
    case kStrided:
        return stream << "Strided";
    case kSparse:
        return stream << "Sparse";
    case kSparseCsr:
        return stream << "SparseCsr";
    case kSparseCsc:
        return stream << "SparseCsc";
    case kSparseBsr:
        return stream << "SparseBsr";
    case kSparseBsc:
        return stream << "SparseBsc";
    default:
        MCHECK(false)<<"Unknown layout";
    }
}

} // namespace mariana

#endif /* __CORE_LAYOUT_H__ */

