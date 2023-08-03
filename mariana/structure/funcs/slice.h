/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/slice.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-03:11:26:04
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_SLICE_H__
#define __STRUCTURE_FUNCS_SLICE_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct SliceOption : public BaseOption {
    SliceOption() {
    }
    ~SliceOption() {
    }
};

struct SliceFunction : public Function {
    SliceFunction() {}
    ~SliceFunction() {}
    SliceOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_SLICE_H__ */

