/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/activation.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:09:43
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_ACTIVATION_H__
#define __STRUCTURE_FUNCS_ACTIVATION_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct ActivationOption : public BaseOption {
    ActivationOption() {}
    ~ActivationOption() {}
};

struct ActivationFunction : public Function {
    ActivationFunction() {}
    ~ActivationFunction() {}
    ActivationOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_ACTIVATION_H__ */

