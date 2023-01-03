/*
 *        (C) COPYRIGHT LeiNao Limited.
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
        std::cout<<"dd:"<<scope.graph_info.tensor_name_map.count(src.input(i))
                 <<" "<<src.input(i)<<std::endl;
        const ::onnx::TensorProto* weight = scope.graph_info.tensor_name_map.at(src.input(i));
        std::vector<int64_t> shape;
        void* content;
        get_content_from_tensor(*weight, shape, &content);
        ArrayRef<int64_t> arr(shape);
        std::cout<<"debug:"<<arr<<std::endl;
        Tensor t;
        t.set_shape(shape);
        ::onnx::TensorProto_DataType data_type = static_cast<::onnx::TensorProto_DataType>(weight->data_type());
        
        switch (data_type) {
        case ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT :
            float* data = t.mutable_data<float>();
            std::cout<<"ss:"<<t.numel()*t.itemsize()<<std::endl;
            memcpy(data, content, t.numel()*t.itemsize());
            
            break;
        }
        
        std::cout<<"use_count "<<t.use_count()<<std::endl;
        std::cout<<::onnx::TensorProto_DataType_Name(data_type)<<"\n";
        func->option.weights.push_back(t);
    }
}

}} // namespace mariana::onnx
