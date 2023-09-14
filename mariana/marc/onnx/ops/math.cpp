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
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void MathConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    MathFunction* func = static_cast<MathFunction*>(dst.op());
    if (src.op_type() == KAdd) {
        func->option.math_type = MathType::kSUM;
    } else if (src.op_type() == KMul) {
        func->option.math_type = MathType::kMUL;
    } else if (src.op_type() == KSub) {
        func->option.math_type = MathType::kSUB;
    } else if (src.op_type() == KDiv) {
        func->option.math_type = MathType::kDIV;
    } else {
        MLOG(FATAL)<<"Unsupport op type:"<<src.op_type();
    }
    int constant = scope.nodes_info.at(src.name()).tensors.empty() ?
        scope.nodes_info.at(src.name()).tensors.size() : 0;
    if (constant != 0) {
        const ::onnx::TensorProto* weight = scope.graph_info.tensor_name_map.at(src.input(0));
        std::vector<int64_t> shape;
        void* content = nullptr;
        get_content_from_tensor(*weight, shape, &content);
        func->option.weight.set_shape(shape);
        ::onnx::TensorProto_DataType data_type = static_cast<::onnx::TensorProto_DataType>(weight->data_type());
        switch (data_type) {
        case ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT : {
            float* data = func->option.weight.mutable_data<float>();
            memcpy(data, content, func->option.weight.numel()*func->option.weight.itemsize());
            break;
        }
        default: {
            MLOG(FATAL)<<"Mar Fatal: unsupport data type in slice.";
        }
        }
    }
}

}} // namespace mariana::onnx
