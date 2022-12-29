/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/flatten.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:30:48
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_FLATTEN_H__
#define __STRUCTURE_FUNCS_FLATTEN_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct FlattenOption : public BaseOption {
    FlattenOption() {}
    ~FlattenOption() {}
};

struct FlattenFunction : public Function {
    FlattenFunction() {}
    ~FlattenFunction() {}
    FlattenOption option;
    tensor_list compute(tensor_list&& inputs) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_FLATTEN_H__ */

