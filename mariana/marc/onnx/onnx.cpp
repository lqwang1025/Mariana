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

#include <set>
#include <fstream>

#include <structure/ir.h>
#include <structure/funcs/register.h>

#include <marc/onnx/ops.h>
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

    graph_info = _init_graph_info(onnx_model.graph());
    nodes_info = _init_nodes_info(onnx_model.graph());
    ::onnx::GraphProto dst;
    sort_by_execution_order(onnx_model.graph(), &dst);
    for (auto it : dst.node()) {
        std::cout<<"node:"<<it.name()<<std::endl;
    }
}

OnnxScope::OnnxScope(const std::string& name) {
    bool ret = this->parse(name, onnx_model);
    MCHECK(ret)<<"Parse onnx model:"<<name<<" failed.";
    _init();
}

static void _sort_by_execution_order(const ::onnx::GraphProto& input_graph,
                                     std::vector<const ::onnx::NodeProto*>& nodes,
                                     std::set<std::string>& visited,
                                     std::unordered_map<std::string, OnnxScope::NodeInfo>& __nodes_info) {
    if (__nodes_info.empty()) return;
    size_t node_size = input_graph.node_size();
    for (size_t i = 0; i < node_size; ++i) {
        if (visited.count(input_graph.node(i).name()) == 1) continue;
        if (__nodes_info[input_graph.node(i).name()].nodes.size() == 0) { // input
            nodes.push_back(&input_graph.node(i));
            visited.insert(input_graph.node(i).name());
            __nodes_info.erase(input_graph.node(i).name());
            
            for (auto& it : __nodes_info) {
                for (size_t _i = 0; _i < it.second.nodes.size(); ++_i) {
                    if (it.second.nodes[_i] == &input_graph.node(i) ||
                        it.second.nodes[_i]->name() == input_graph.node(i).name()) {
                        it.second.nodes.erase(it.second.nodes.begin()+_i);
                        break;
                    }
                }
            }
        }
    }
    _sort_by_execution_order(input_graph, nodes, visited, __nodes_info);
}

std::unordered_map<std::string, OnnxScope::NodeInfo> OnnxScope::_init_nodes_info(
    const ::onnx::GraphProto& graph) {
    GraphInfo _graph_info = _init_graph_info(graph);
    std::unordered_map<std::string, NodeInfo> _nodes_info;
    for (const ::onnx::NodeProto& node : onnx_model.graph().node()) {
        NodeInfo node_info;
        for (auto& input : node.input()) {
            if (_graph_info.tensor_name_map.count(input) != 0) {// Input of node is tensor.
                node_info.tensors.push_back(_graph_info.tensor_name_map[input]);
            } else { // Input of node is node.
                for (const ::onnx::NodeProto& _node : graph.node()) {
                    for (auto& _output : _node.output()) {
                        if (input == _output) {
                            node_info.nodes.push_back(const_cast<::onnx::NodeProto*>(&_node));
                            break;
                        }
                    }
                }
            }
        }
        _nodes_info.emplace(node.name(), node_info);
    }
    return _nodes_info;
}

OnnxScope::GraphInfo OnnxScope::_init_graph_info(const ::onnx::GraphProto& graph) {
    GraphInfo _graph_info;
    // graph info initilized
    for (auto& it : graph.node()) {
        _graph_info.node_name_map.insert({it.name(), const_cast<::onnx::NodeProto*>(&it)});
    }

    for (auto& it : graph.initializer()) {
        _graph_info.tensor_name_map.insert({it.name(), const_cast<::onnx::TensorProto*>(&it)});
    }
    _graph_info.graph = const_cast<::onnx::GraphProto*>(&graph);
    return _graph_info;
}

Status OnnxScope::sort_by_execution_order(const ::onnx::GraphProto& input_graph,
                                          ::onnx::GraphProto* output_graph) {
    std::vector<const ::onnx::NodeProto*> order_nodes(0);
    std::set<std::string> visited;
    size_t node_size = input_graph.node_size();
    std::unordered_map<std::string, NodeInfo> __nodes_info = _init_nodes_info(input_graph);
    _sort_by_execution_order(input_graph, order_nodes, visited, __nodes_info);
    output_graph->CopyFrom(input_graph);
    output_graph->clear_node();
    for (auto it : order_nodes) {
        *output_graph->add_node() = *it;
    }
}

Graph* parse(const std::string& name) {
    register_converter();
    OnnxScope onnx_scope(name);
    Graph* graph = new Graph{};
    // register_funcs();
    // for (const ::onnx::NodeProto& node : onnx_scope.graph_info.graph->node()) {
    //     if (1 == CONTINUE_OP.count(node.op_type())) continue;
    //     OnnxConverter* convert = OnnxHolder::search(node.op_type());
    //     Node& dst = graph->add_node(node.name(), node.op_type());
    //     convert->run(node, dst, onnx_scope);
    // }
    // unregister_funcs();
    // unregister_converter();
    return graph;
}

}} // namespace mariana::onnx
