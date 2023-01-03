/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/gemm.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:30:48
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_GEMM_H__
#define __STRUCTURE_FUNCS_GEMM_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

struct GemmOption : public BaseOption {
    GemmOption() {}
    ~GemmOption() {}
    float alpha = 1.f;
    float beta = 1.f;
    bool trans_a = false;
    bool trans_b = false;
    std::vector<Tensor> weights;
};

struct GemmFunction : public Function {
    GemmFunction() {}
    ~GemmFunction() {}
    GemmOption option;
    tensor_list compute(tensor_list&& inputs) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_GEMM_H__ */

