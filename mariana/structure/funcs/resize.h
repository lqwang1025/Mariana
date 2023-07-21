/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/resize.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:15:44:55
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_RESIZE_H__
#define __STRUCTURE_FUNCS_RESIZE_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct ResizeOption : public BaseOption {
    ResizeOption() {}
    ~ResizeOption() {}
};

struct ResizeFunction : public Function {
    ResizeFunction() {}
    ~ResizeFunction() {}
    ResizeOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_RESIZE_H__ */

