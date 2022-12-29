/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/relu.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:09:43
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_RELU_H__
#define __STRUCTURE_FUNCS_RELU_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct ReluOption : public BaseOption {
    ReluOption() {}
    ~ReluOption() {}
};

struct ReluFunction : public Function {
    ReluFunction() {}
    ~ReluFunction() {}
    ReluOption option;
    tensor_list compute(tensor_list&& inputs) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_RELU_H__ */

