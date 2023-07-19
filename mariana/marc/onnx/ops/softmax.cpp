/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : softmax.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:16:22:39
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/softmax.h>
#include <marc/onnx/proto/onnx_help.h>
#include <iostream>

namespace mariana { namespace onnx {

void SoftmaxConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    SoftmaxFunction* func = static_cast<SoftmaxFunction*>(dst.op());
    int64_t axis = 0;
    GET_ONNX_NODE_ATTR(src, "axis", &axis);
    func->option.axis = static_cast<uint32_t>(axis);
}

}} // namespace mariana::onnx

