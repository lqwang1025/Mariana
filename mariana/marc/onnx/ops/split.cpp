/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : split.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:24
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/split.h>
#include <marc/onnx/proto/onnx_help.h>
#include <core/utils/logging.h>

namespace mariana { namespace onnx {

void SplitConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    SplitFunction* func = static_cast<SplitFunction*>(dst.op());
    GET_ONNX_NODE_ATTR(src, "axis", &func->option.axis);
    GET_ONNX_NODE_ATTR(src, "split", &func->option.split);
}

}} // namespace mariana::onnx

