/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : conv.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:51:29
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/conv.h>
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void ConvConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    ConvFunction* func = static_cast<ConvFunction*>(dst.op());
    
    GET_ONNX_NODE_ATTR(src, "dilations", &func->option.dilations);
    GET_ONNX_NODE_ATTR(src, "group", &func->option.group);
    GET_ONNX_NODE_ATTR(src, "kernel_shape", &func->option.kernel_shape);
    GET_ONNX_NODE_ATTR(src, "strides", &func->option.strides);
    GET_ONNX_NODE_ATTR(src, "pads", &func->option.pads);
    
    func->option.weights.reserve(scope.nodes_info.at(src.name()).tensors.size());
    
    for (int i = 1; i < src.input_size(); ++i) {
        const ::onnx::TensorProto* weight = scope.graph_info.tensor_name_map.at(src.input(i));
        std::vector<int64_t> shape;
        void* content = nullptr;
        get_content_from_tensor(*weight, shape, &content);
        Tensor t;
        t.set_shape(shape);
        ::onnx::TensorProto_DataType data_type = static_cast<::onnx::TensorProto_DataType>(weight->data_type());
        
        switch (data_type) {
        case ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT : {
            float* data = t.mutable_data<float>();
            memcpy(data, content, t.numel()*t.itemsize());
            break;
        }
        default: {
            MLOG(FATAL)<<"Mar Fatal: unsupport data type in slice.";
        }
        }
        func->option.weights.push_back(t);
    }
    func->option.oc = func->option.weights[0].shape()[0];
}

}} // namespace mariana::onnx
