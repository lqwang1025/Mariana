/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/onnx.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-26:17:42:39
 * Description:
 * 
 */

#include <fstream>

#include <marc/onnx/onnx.h>
#include <marc/onnx/register.h>
#include <core/utils/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace mariana { namespace onnx {

bool OnnxScope::parse(const std::string& name, ::onnx::ModelProto& onnx_model) {
    std::ifstream is(name, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        MCHECK(false)<<"Open file:"<<name<<" failed.";
    }
    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);

    coded_input.SetTotalBytesLimit(1024 << 20);

    bool ret = onnx_model.ParseFromCodedStream(&coded_input);
    is.close();
    return ret;
}

void OnnxScope::_init() {
    model_info.ir_version = onnx_model.ir_version();
    model_info.producer_name = onnx_model.producer_name();
    model_info.producer_version = onnx_model.producer_version();
    model_info.domain = onnx_model.domain();
    model_info.model_version = onnx_model.model_version();
    model_info.doc_string = onnx_model.doc_string();
    
    // graph info initilized
    for (auto it : onnx_model.graph().node()) {
        graph_info.node_name_map.insert({it.name(), &it});
    }
    std::cout<<"size:"<<graph_info.node_name_map.size()<<std::endl;
    for (auto it : onnx_model.graph().initializer()) {
        graph_info.tensor_name_map.insert({it.name(), &it});
    }
    graph_info.graph = onnx_model.mutable_graph();
    std::cout<<"size:"<<graph_info.tensor_name_map.size()<<std::endl;
}

OnnxScope::OnnxScope(const std::string& name) {
    bool ret = this->parse(name, onnx_model);
    MCHECK(ret)<<"Parse onnx model:"<<name<<" failed.";
    _init();
}

bool parse(const std::string& name) {
    register_converter();
    OnnxScope onnx_scope(name);
    for (const ::onnx::NodeProto& node : onnx_scope.graph_info.graph->node()) {
        OnnxConverter* convert = OnnxHolder::search(node.op_type());
        convert->run(node, onnx_scope);
    }
    unregister_converter();
}

}} // namespace mariana::onnx
