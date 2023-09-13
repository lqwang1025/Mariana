/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : transpose.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:33
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/permute.h>
#include <marc/onnx/proto/onnx_help.h>
#include <core/utils/logging.h>

namespace mariana { namespace onnx {

void TransposeConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    PermuteFunction* func = static_cast<PermuteFunction*>(dst.op());
    GET_ONNX_NODE_ATTR(src, "perm", &func->option.perm);
}

}} // namespace mariana::onnx

