/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/register.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:47:33
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <core/macros/macros.h>

namespace mariana { namespace onnx {

#define ADD_CONVERT(identity, type)                                     \
    OnnxHolder::add_onnx_convert(std::move(MAR_STRINGIZE(identity)),    \
                                 std::move(new type{}))
void register_converter() {
    ADD_CONVERT(Default, DefaultConverter);
    ADD_CONVERT(Add, AddConverter);
    ADD_CONVERT(Concat, ConcatConverter);
    ADD_CONVERT(MaxPool, MaxPoolConverter);
    ADD_CONVERT(Mul, MulConverter);
    ADD_CONVERT(Pow, PowConverter);
    ADD_CONVERT(Conv, ConvConverter);
    ADD_CONVERT(Reshape, ReshapeConverter);
    ADD_CONVERT(Resize, ResizeConverter);
    ADD_CONVERT(Sigmoid, SigmoidConverter);
    ADD_CONVERT(Split, SplitConverter);
    ADD_CONVERT(Transpose, TransposeConverter);
    ADD_CONVERT(Relu, ReluConverter);
}
#undef ADD_CONVERT

void unregister_converter() {
    OnnxHolder::release();
}

}} // namespace mariana::onnx
