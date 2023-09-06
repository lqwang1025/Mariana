/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/split.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-20:17:04:44
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_SPLIT_H__
#define __STRUCTURE_FUNCS_SPLIT_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct SplitOption : public BaseOption {
    SplitOption() {}
    ~SplitOption() {}
    int32_t axis = -1;
    std::vector<int32_t> split;
};

struct SplitFunction : public Function {
    SplitFunction() {}
    ~SplitFunction() {}
    SplitOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_SPLIT_H__ */

