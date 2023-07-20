/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/math.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:28:38
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_MATH_H__
#define __STRUCTURE_FUNCS_MATH_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct MathOption : public BaseOption {
    MathOption() {}
    ~MathOption() {}
};

struct MathFunction : public Function {
    MathFunction() {}
    ~MathFunction() {}
    MathOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_MATH_H__ */

