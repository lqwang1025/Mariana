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

detail::TypeMetaData* TypeMeta::type_meta_datas() {
    static detail::TypeMetaData instances[UINT8_MAX+1] = {
        detail::TypeMetaData(TypeUninitIndex, 0, "Uninit"),
#define SCALAR_TYPE_META(T, idx)                                \
        detail::TypeMetaData(idx, sizeof(T), MAR_STRINGIZE(T)),
        FORALL_SCALAR_TYPES(SCALAR_TYPE_META)
    };
#undef SCALAR_TYPE_META
    return instances;
};
FORALL_SCALAR_TYPES(DELECARE_META_DATA);
} // namespace mariana
