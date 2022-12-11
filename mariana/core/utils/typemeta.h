/*
 * (C) COPYRIGHT Daniel Wang Limited.
 * File       : core/utils/typemeta.h
 * Authors    : lqwang@Pandora-2.local
 * Create Time: 2022-12-11:10:26:23
 * Email      : wangliquan21@qq.com
 * Description:
 * 
 */

#ifndef _CORE_UTILS_TYPEMETA_H_
#define _CORE_UTILS_TYPEMETA_H_

#include <string>
#define MaxTypeIndex UINT8_MAX

namespace detail {

struct TypeMetaData {
    TypeMetaData(int32_t idx, size_t is, std::string n) : index(idx), itemsize(is), name(n) {}
    ~TypeMetaData() {}
    int32_t index;
    size_t itemsize;
    std::string name;
};

} // namespace detial

namespace mariana {

static detail::TypeMetaData* typeMetaDatas();

struct TypeMeta final {
    static TypeMeta make() {
        
    }
};

} // namespace mariana

#endif /* _CORE_UTILS_TYPEMETA_H_ */

