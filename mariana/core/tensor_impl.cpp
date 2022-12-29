/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_impl.cpp
 * Authors    : lqwang@pandora
 * Create Time: 2022-11-29:20:49:22
 * Description:
 * 
 */

#include <core/tensor_impl.h>
#include <core/macros/macros.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

void TensorImpl::reshape(IntArrayRef shape) {
    
}

void TensorImpl::share_data(const TensorImpl& src) {
    
}

void TensorImpl::ShareExternalPointer(DataPtr&& data_ptr, TypeMeta data_type, size_t size_bytes) {
    
}

} // namespace mariana
