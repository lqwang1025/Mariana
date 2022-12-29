/*
 *        (C) COPYRIGHT LeiNao Limited.
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
#include <structure/funcs/relu.h>
#include <structure/funcs/pool.h>
#include <structure/funcs/gemm.h>
#include <structure/funcs/add.h>
#include <structure/funcs/flatten.h>

namespace mariana {

#define ADD_FUNC(identity, type)                                        \
    static auto __##identity##_make = []()->Function* { return new type{}; }; \
    FunctionHolder::add_func(std::move(MAR_STRINGIZE(identity)),        \
                             __##identity##_make)

void register_funcs() {
    ADD_FUNC(Conv, ConvFunction);
    ADD_FUNC(Relu, ReluFunction);
    ADD_FUNC(MaxPool, PoolFunction);
    ADD_FUNC(GlobalAveragePool, PoolFunction);
    ADD_FUNC(Add, AddFunction);
    ADD_FUNC(Gemm, GemmFunction);
    ADD_FUNC(Flatten, FlattenFunction);
}
#undef ADD_FUNC

void unregister_funcs() {
    FunctionHolder::release();
}

} // namespace mariana {
