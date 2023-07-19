/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/softmax.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:16:07:03
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_SOFTMAX_H__
#define __STRUCTURE_FUNCS_SOFTMAX_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct SoftmaxOption : public BaseOption {
    SoftmaxOption() {}
    ~SoftmaxOption() {}
    uint32_t axis = 0;
};

struct SoftmaxFunction : public Function {
    SoftmaxFunction() {}
    ~SoftmaxFunction() {}
    SoftmaxOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana


#endif /* __STRUCTURE_FUNCS_SOFTMAX_H__ */

