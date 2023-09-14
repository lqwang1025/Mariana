/*
 *        (C) COPYRIGHT Daniel Limited.
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
    ADD_CONVERT(Add, MathConverter);
    ADD_CONVERT(Mul, MathConverter);
    ADD_CONVERT(Sub, MathConverter);
    ADD_CONVERT(Div, MathConverter);
    ADD_CONVERT(Concat, ConcatConverter);
    ADD_CONVERT(MaxPool, PoolConverter);
    ADD_CONVERT(GlobalAveragePool, PoolConverter);
    ADD_CONVERT(Pow, PowConverter);
    ADD_CONVERT(Conv, ConvConverter);
    ADD_CONVERT(Reshape, ReshapeConverter);
    ADD_CONVERT(Resize, ResizeConverter);
    ADD_CONVERT(Sigmoid, ActConverter);
    ADD_CONVERT(Split, SplitConverter);
    ADD_CONVERT(Transpose, TransposeConverter);
    ADD_CONVERT(Relu, ActConverter);
    ADD_CONVERT(Gemm, GemmConverter);
    ADD_CONVERT(Softmax, SoftmaxConverter);
    ADD_CONVERT(ReduceMean, ReduceConverter);
    ADD_CONVERT(Flatten, FlattenConverter);
    ADD_CONVERT(Slice, SliceConverter);
}
#undef ADD_CONVERT

void unregister_converter() {
    OnnxHolder::release();
}

}} // namespace mariana::onnx
