/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : relu.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:38
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/relu.h>

namespace mariana { namespace onnx {

void ReluConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
}

}} // namespace mariana::onnx

