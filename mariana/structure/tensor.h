/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/tensor.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:11:20
 * Description:
 *
 */

#ifndef __STRUCTURE_TENSOR_H__
#define __STRUCTURE_TENSOR_H__

#include <core/tensor_impl.h>
#include <core/utils/logging.h>

namespace mariana {

class Tensor {
public:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
        if (impl_.get() == nullptr) {
            //TODO:ERROR
        }
    }
    Tensor(const Tensor&)=default;
    Tensor(Tensor&&)=default;
    uint8_t dim() const {
        return impl_->dim();
    }
    int64_t numel() const {
        return impl_->numel();
    }
    int64_t storage_offset() const {
        return impl_->storage_offset();
    }
    
private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace mariana

#endif /* __STRUCTURE_TENSOR_H__ */

