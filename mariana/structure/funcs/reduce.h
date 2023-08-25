/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/reduce.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-25:08:59:53
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_REDUCE_H__
#define __STRUCTURE_FUNCS_REDUCE_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

enum class ReduceMethod : int8_t {
    UNINIT = -1,
    SUM    = 0,
    MEAN   = 1,
};

struct ReduceOption : public BaseOption {
    ReduceOption() {
        axes.clear();
    }
    ~ReduceOption() {}
    ReduceMethod method = ReduceMethod::UNINIT;
    bool keepdims = false;
    std::vector<int32_t> axes;
};

struct ReduceFunction : public Function {
    ReduceFunction() {}
    ~ReduceFunction() {}
    ReduceOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_REDUCE_H__ */

