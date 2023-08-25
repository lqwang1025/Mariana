/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops/reduce.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-25:09:23:13
 * Description:
 * 
 */

#include <marc/onnx/ops.h>
#include <core/utils/logging.h>
#include <marc/onnx/register.h>
#include <structure/funcs/reduce.h>
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void ReduceConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& onnx_scope) {
    ReduceFunction* func = static_cast<ReduceFunction*>(dst.op());
    if (src.op_type() == kReduceMean) {
        func->option.method = ReduceMethod::MEAN;
    } else {
        MLOG(FATAL)<<"Unsupport op type:"<<src.op_type();
    }

    GET_ONNX_NODE_ATTR(src, "axes", &func->option.axes);
    int64_t keepdims = 0;
    GET_ONNX_NODE_ATTR(src, "keepdims", &keepdims);
    if (keepdims) {
        func->option.keepdims = true;
    }
}

}} // namespace mariana::onnx

