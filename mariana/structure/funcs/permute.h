/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/permute.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:16:07:39
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_PERMUTE_H__
#define __STRUCTURE_FUNCS_PERMUTE_H__
#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct PermuteOption : public BaseOption {
    PermuteOption() {}
    ~PermuteOption() {}
};

struct PermuteFunction : public Function {
    PermuteFunction() {}
    ~PermuteFunction() {}
    PermuteOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_PERMUTE_H__ */

