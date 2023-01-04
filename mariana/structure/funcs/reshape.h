/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/reshape.h
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:10:34:51
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_RESHAPE_H__
#define __STRUCTURE_FUNCS_RESHAPE_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct ReshapeOption : public BaseOption {
    ReshapeOption() {}
    ~ReshapeOption() {}
};

struct ReshapeFunction : public Function {
    ReshapeFunction() {}
    ~ReshapeFunction() {}
    ReshapeOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_RESHAPE_H__ */

