/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops/math.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:51:11
 * Description:
 * 
 */

#include <marc/onnx/ops.h>
#include <core/utils/logging.h>
#include <marc/onnx/register.h>
#include <structure/funcs/math.h>

namespace mariana { namespace onnx {

void MathConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    MathFunction* func = static_cast<MathFunction*>(dst.op());
    if (src.op_type() == KAdd) {
        func->option.math_type = MathType::kSUM;
    } else if (src.op_type() == KMul) {
        func->option.math_type = MathType::kMUL;
    } else {
        MLOG(FATAL)<<"Unsupport op type:"<<src.op_type();
    }

    
    
}

}} // namespace mariana::onnx
