/*
 *        (C) COPYRIGHT Daniel Limited.
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
#include <marc/onnx/optimize/transform.h>

#include <core/utils/logging.h>
#include <maro/transform_utils.h>
#include <marc/onnx/optimize/transform_utils.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace mariana { namespace onnx {

bool OnnxScope::save(const std::string& name) {
    std::ofstream fs(name);
    if (fs.fail()) {
        MCHECK(false)<<"Open file:"<<name<<" failed.";
        return false;
    }
    onnx_model.SerializeToOstream(&fs);
    fs.close();
    return true;
}

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
    
    graph_info = init_graph_info(onnx_model.graph());
    nodes_info = init_nodes_info(onnx_model.graph());
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
        const std::string& node_name = input_graph.node(i).name();
        if (visited.count(node_name) == 1) continue;
        if (__nodes_info.at(node_name).nodes.size() == 0) { // input
            nodes.push_back(&input_graph.node(i));
            visited.insert(node_name);
            __nodes_info.erase(node_name);
            
            for (auto& it : __nodes_info) {
                for (auto iter = it.second.nodes.begin(); iter != it.second.nodes.end(); ++iter) {
                    if (*iter == &input_graph.node(i) || (*iter)->name() == node_name) {
                        iter = it.second.nodes.erase(iter);
                        iter--;
                    }
                }
            }
        }
    }
    _sort_by_execution_order(input_graph, nodes, visited, __nodes_info);
}

std::unordered_map<std::string, OnnxScope::NodeInfo> OnnxScope::init_nodes_info(
    const ::onnx::GraphProto& graph) {
    GraphInfo _graph_info = init_graph_info(graph);
    std::unordered_map<std::string, NodeInfo> _nodes_info;
    for (const ::onnx::NodeProto& node : graph.node()) {
        NodeInfo node_info;
        for (auto& input : node.input()) {
            std::string input_name;
            if (_graph_info.tensor_name_map.count(input) != 0) {// Input of node is tensor.
                node_info.tensors.push_back(_graph_info.tensor_name_map[input]);
                input_name = input;
            } else { // Input of node is node.
                bool finded = false;
                for (const ::onnx::NodeProto& _node : graph.node()) {
                    if (finded) break;
                    for (auto& _output : _node.output()) {
                        if (input == _output) {
                            node_info.nodes.push_back(const_cast<::onnx::NodeProto*>(&_node));
                            input_name = _node.name();
                            finded = true;
                            break;
                        }
                    }
                }
            }
            node_info.inputs.push_back(input_name);
        }
        _nodes_info.emplace(node.name(), node_info);
    }
    return _nodes_info;
}

OnnxScope::GraphInfo OnnxScope::init_graph_info(const ::onnx::GraphProto& graph) {
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
    std::unordered_map<std::string, NodeInfo> __nodes_info = init_nodes_info(input_graph);
    _sort_by_execution_order(input_graph, order_nodes, visited, __nodes_info);
    output_graph->CopyFrom(input_graph);
    output_graph->clear_node();
    for (auto it : order_nodes) {
        *output_graph->add_node() = *it;
    }
    return absl::OkStatus();
}

void OnnxScope::update(const ::onnx::GraphProto& graph) {
    sort_by_execution_order(graph, onnx_model.mutable_graph());
    graph_info = init_graph_info(onnx_model.graph());
    nodes_info = init_nodes_info(onnx_model.graph());
}

Graph* parse(const std::string& name) {
    register_converter();
    OnnxScope onnx_scope(name);

    // 1. Optimie onnx graph first.
    transform::transform(onnx_scope, {"fold_identity_to_conv"});
    
    // 2. Convert onnx node to us. The node had been sorted.
    Graph* graph = new Graph{};
    register_funcs();
    for (const ::onnx::NodeProto& node : onnx_scope.graph_info.graph->node()) {
        OnnxConverter* convert = OnnxHolder::search(node.op_type());
        MCHECK(ONNX_OP_MAP_TO_MAR.count(node.op_type()))<<"Mar is not support :[OPTYPE:"
                                                        <<node.op_type()<<"].";
        const std::string& op_type = ONNX_OP_MAP_TO_MAR.at(node.op_type());
        Node& dst = graph->new_node(node.name(), op_type);
        convert->run(node, dst, onnx_scope);
        // 3. Build link on our graph.
        auto& inodes = onnx_scope.nodes_info.at(node.name()).nodes; // Inputs
        
        std::set<std::string> inputs;
        for (size_t i = 0; i < node.input_size(); ++i) {
            inputs.insert(node.input(i));
        }
        for (size_t i = 0; i < inodes.size(); ++i) { // input
            size_t ii = 0;
            for (; ii < inodes[i]->output_size(); ++ii) {
                auto fit = inputs.find(inodes[i]->output(ii));
                if (fit != inputs.end()) {
                    inputs.erase(fit);
                    break;
                }
            }
            dst.inputs().push_back(inodes[i]->name());
            dst.ctrl_idx().push_back(ii);
        }
    }
    graph->finilize();
    
    unregister_funcs();
    unregister_converter();
    return graph;
}

}} // namespace mariana::onnx
