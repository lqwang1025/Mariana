/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : maxpool.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:51:38
 * Description:
 * 
 */

#include <structure/funcs/pool.h>
#include <core/utils/logging.h>

#include <marc/onnx/ops.h>
#include <marc/onnx/register.h>
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void PoolConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope){ 
    PoolFunction* func = static_cast<PoolFunction*>(dst.op());
    if (src.op_type() == KMaxPool) {
        func->option.type = PoolType::Max;
        GET_ONNX_NODE_ATTR(src, "kernel_shape", &func->option.kernel_shape);
        GET_ONNX_NODE_ATTR(src, "pads", &func->option.pads);
        GET_ONNX_NODE_ATTR(src, "pads", &func->option.strides);
        GET_ONNX_NODE_ATTR(src, "ceil_mode", &func->option.ceil_mode);
    } else if (src.op_type() == KGlobalAveragePool) {
        func->option.type = PoolType::GAvg;
    } else {
        MCHECK(false)<<"Unsupport pool type:"<<src.op_type();
    }
}

}} // namespace mariana::onnx
