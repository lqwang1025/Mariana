/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops/concat.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:51:20
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/concat.h>
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void ConcatConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    ConcatFunction* func = static_cast<ConcatFunction*>(dst.op());
    GET_ONNX_NODE_ATTR(src, "axis", &func->option.axis);
}

}} // namespace mariana::onnx
