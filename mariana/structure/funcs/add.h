/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/add.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:28:38
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_ADD_H__
#define __STRUCTURE_FUNCS_ADD_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct AddOption : public BaseOption {
    AddOption() {}
    ~AddOption() {}
};

struct AddFunction : public Function {
    AddFunction() {}
    ~AddFunction() {}
    AddOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_ADD_H__ */

