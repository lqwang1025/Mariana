/*
 * (C) COPYRIGHT Daniel Wang Limited.
 * File       : core/utils/typemeta.cpp
 * Authors    : lqwang@Pandora-2.local
 * Create Time: 2022-12-11:11:16:51
 * Email      : wangliquan21@qq.com
 * Description:
 * 
 */

#include <cstdint>
#include <core/macros/macros.h>
#include <core/utils/typemeta.h>

namespace mariana {

#define FORALL_SCALAR_TYPES(_)                  \
    _(uint8_t, 0)                               \
    _(int8_t, 1)                                \
    _(int16_t, 2)                               \
    _(int32_t, 3)                               \
    _(uint32_t, 4)                              \
    _(int64_t, 5)                               \
    _(float, 6)                                 \
    _(double, 7)                                \
    _(bool, 8)                           

detail::TypeMetaData* TypeMeta::datas() {
    static detail::TypeMetaData instances[UINT8_MAX+1] = {
#define SCALAR_TYPE_META(T, idx)                                \
        detail::TypeMetaData(idx, sizeof(T), MAR_STRINGIZE(T)),
        FORALL_SCALAR_TYPES(SCALAR_TYPE_META)
    };
#undef SCALAR_TYPE_META
    return instances;
};

} // namespace mariana
