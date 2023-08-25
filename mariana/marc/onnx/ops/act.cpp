/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : act.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:38
 * Description:
 * 
 */

#include <marc/onnx/ops.h>
#include <core/utils/logging.h>
#include <marc/onnx/register.h>
#include <structure/funcs/activation.h>

namespace mariana { namespace onnx {

void ActConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    ActivationFunction* func = static_cast<ActivationFunction*>(dst.op());
    if (src.op_type() == KRelu) {
        func->option.act_type = ActivationType::kRELU;
    } else if (src.op_type() == KSigmoid) {
        func->option.act_type = ActivationType::kSIGMOID;
    } else {
        MLOG(FATAL)<<"Unsupport op type:"<<src.op_type();
    }
    
}

}} // namespace mariana::onnx

