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

namespace mariana { namespace detail {

struct TypeMetaData {
    TypeMetaData(int32_t idx, size_t is, const std::string& n) : index(idx), itemsize(is), name(n) {}
    TypeMetaData() : index(-1), itemsize(0), name("nullptr Uninitalized") {}
    ~TypeMetaData() {}
    int32_t index;
    size_t itemsize;
    std::string name;
};

} // namespace detail

struct TypeMeta final {
    static TypeMeta make() {
        
    }
    static detail::TypeMetaData* datas();
};

} // namespace mariana

#endif /* _CORE_UTILS_TYPEMETA_H_ */

