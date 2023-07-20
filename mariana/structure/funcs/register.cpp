/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/register.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:10:10:30
 * Description:
 * 
 */

#include <core/macros/macros.h>
#include <structure/funcs/register.h>
#include <structure/function.h>
#include <structure/funcs/conv.h>
#include <structure/funcs/activation.h>
#include <structure/funcs/pool.h>
#include <structure/funcs/gemm.h>
#include <structure/funcs/math.h>
#include <structure/funcs/flatten.h>
#include <structure/funcs/reshape.h>
#include <structure/funcs/softmax.h>

namespace mariana {

#define ADD_FUNC(identity, type)                                        \
    static auto __##identity##_make = []()->Function* { return new type{}; }; \
    FunctionHolder::add_func(std::move(MAR_STRINGIZE(identity)),        \
                             __##identity##_make)

void register_funcs() {
    ADD_FUNC(CONV2D, ConvFunction);
    ADD_FUNC(RELU, ActivationFunction);
    ADD_FUNC(SIGMOID, ActivationFunction);
    ADD_FUNC(SOFTMAX, SoftmaxFunction);
    ADD_FUNC(RESHAPE, ReshapeFunction);
    ADD_FUNC(MAXPOOL, PoolFunction);
    ADD_FUNC(GAVPOOL, PoolFunction);
    ADD_FUNC(ADD, MathFunction);
    ADD_FUNC(MUL, MathFunction);
    ADD_FUNC(GEMM, GemmFunction);
    ADD_FUNC(FLATTEN, FlattenFunction);
}
#undef ADD_FUNC

void unregister_funcs() {
    FunctionHolder::release();
}

} // namespace mariana {
