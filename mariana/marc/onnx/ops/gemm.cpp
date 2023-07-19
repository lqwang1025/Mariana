/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops/gemm.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-03:13:58:10
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/gemm.h>
#include <marc/onnx/proto/onnx_help.h>
namespace mariana { namespace onnx {

void GemmConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    GemmFunction* func = static_cast<GemmFunction*>(dst.op());
    GET_ONNX_NODE_ATTR(src, "alpha", &func->option.alpha);
    GET_ONNX_NODE_ATTR(src, "beta", &func->option.beta);
    int32_t trans = 0;
    if (node_has_attr(src, "transA")) {
        GET_ONNX_NODE_ATTR(src, "transA", &trans);
    }
    if (trans == 1) {
        func->option.trans_a = true;
        trans = 0;
    }
    if (node_has_attr(src, "transB")) {
        GET_ONNX_NODE_ATTR(src, "transB", &trans);
    }
    if (trans == 1) {
        func->option.trans_b = true;
    }
    func->option.weights.reserve(scope.nodes_info.at(src.name()).tensors.size());
    for (int i = 1; i < src.input_size(); ++i) {
        const ::onnx::TensorProto* weight = scope.graph_info.tensor_name_map.at(src.input(i));
        std::vector<int64_t> shape;
        void* content;
        get_content_from_tensor(*weight, shape, &content);
        Tensor t;
        t.set_shape(shape);
        ::onnx::TensorProto_DataType data_type = static_cast<::onnx::TensorProto_DataType>(weight->data_type());
        
        switch (data_type) {
        case ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT :
            float* data = t.mutable_data<float>();
            memcpy(data, content, t.numel()*t.itemsize());
            break;
        }
        func->option.weights.push_back(t);
    }
}

}} // namespace mariana::onnx
