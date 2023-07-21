/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/concat.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-20:17:32:57
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_CONCAT_H__
#define __STRUCTURE_FUNCS_CONCAT_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct ConcatOption : public BaseOption {
    ConcatOption() {
    }
    ~ConcatOption() {
    }
};

struct ConcatFunction : public Function {
    ConcatFunction() {}
    ~ConcatFunction() {}
    ConcatOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_CONCAT_H__ */

