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
    
    func->option.weights.reserve(src.input_size()-1);
    
    // for (int i = 1; i < src.input_size(); ++i) {
    //     std::cout<<"dd:"<<scope.graph_info.node_name_map.count(src.input(i))
    //              <<" "<<src.input(i)<<std::endl;
    //     std::cout<<"dd:"<<scope.graph_info.tensor_name_map.count(src.input(i))
    //              <<" "<<src.input(i)<<std::endl;
        // Tensor t;
        // t.set_shape(IntArrayRef shape, int64_t storage_offset = 0);
        // func->option.weights.push_back();
    // }
    for (auto it : scope.nodes_info.at(src.name()).nodes) {
        std::cout<<"input node:"<<it->name()<<std::endl;
    }
    std::cout<<"debug:"<<src.name()<<" "<<scope.nodes_info.at(src.name()).tensors.size()<<std::endl;
    for (auto it : scope.nodes_info.at(src.name()).tensors) {
        std::cout<<"input t:"<<it->name()<<std::endl;
    }
}

}} // namespace mariana::onnx
