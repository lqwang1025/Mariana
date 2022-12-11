/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_impl.h
 * Authors    : lqwang@pandora
 * Create Time: 2022-11-29:20:49:13
 * Description:
 *
 */

#ifndef __CORE_TENSOR_IMPL_H__
#define __CORE_TENSOR_IMPL_H__

#include <core/storage.h>
#include <core/utils/typemeta.h>

namespace mariana {

class TensorImp {
public:
    TensorImp(Storage&& storge, const TypeMeta data_type);
    ~TensorImp();
private:
    TypeMeta data_type_;
    Storage storage_;
    int64_t nnumel_;
};

} // namespace mariana
#endif /* __CORE_TENSOR_IMPL_H__ */

