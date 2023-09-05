/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : flatten.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-09-05:15:16:41
 * Description:
 * 
 */

#include <marc/onnx/ops.h>
#include <core/utils/logging.h>
#include <marc/onnx/register.h>
#include <structure/funcs/flatten.h>
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void FlattenConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    FlattenFunction* func = static_cast<FlattenFunction*>(dst.op());
    int64_t axis = 0;
    GET_ONNX_NODE_ATTR(src, "axis", &axis);
    func->option.axis = static_cast<int32_t>(axis);
}

}} // namespace mariana::onnx
