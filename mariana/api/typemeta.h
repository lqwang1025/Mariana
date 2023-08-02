/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : typemeta.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:09:19:45
 * Description:
 *
 */

#ifndef __TYPEMETA_H__
#define __TYPEMETA_H__

#include <string>
#define MaxTypeIndex UINT8_MAX
#define TypeUninitIndex -1
#define FORALL_SCALAR_TYPES(_)                  \
    _(uint8_t, 1)                               \
    _(int8_t, 2)                                \
    _(int16_t, 3)                               \
    _(int32_t, 4)                               \
    _(uint32_t, 5)                              \
    _(int64_t, 6)                               \
    _(float, 7)                                 \
    _(double, 8)                                \
    _(bool, 9)

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
    inline TypeMeta() : index_(TypeUninitIndex) {}
    template<typename T>
    static TypeMeta make() {
        return TypeMeta(_type_meta_data<T>());
    }
    template<typename T>
    bool match() const {
        return (*this == make<T>());
    }
    static detail::TypeMetaData* type_meta_datas();
    inline size_t itemsize() const {
        return data().itemsize;
    }
    detail::TypeMetaData data() const {
        return type_meta_datas()[index_];
    }
    friend bool operator==(const TypeMeta lhs, const TypeMeta rhs);
    friend bool operator!=(const TypeMeta lhs, const TypeMeta rhs);
private:
    TypeMeta(const uint16_t index) : index_(index) {}
    template <typename T>
    static uint16_t _type_meta_data();
    uint16_t index_;
};

inline bool operator==(const TypeMeta lhs, const TypeMeta rhs) {
    return (lhs.index_ == rhs.index_);
}

inline bool operator!=(const TypeMeta lhs, const TypeMeta rhs) {
    return !operator==(lhs, rhs);
}

#define DELECARE_META_DATA(T, idx)                  \
    template<>                                      \
    uint16_t TypeMeta::_type_meta_data<T>() {        \
        return idx;                                  \
    }

} // namespace mariana

#endif /* __TYPEMETA_H__ */

