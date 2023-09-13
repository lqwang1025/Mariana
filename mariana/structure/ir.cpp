/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/ir.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-28:08:55:41
 * Description:
 * 
 */

#include <structure/ir.h>
#include <core/utils/logging.h>

namespace mariana {

Node::EdgeEnd::EdgeEnd(std::shared_ptr<Node>& node, int index) noexcept : node_(node.get()),
                                                                          index_(index) {}

Node::EdgeEnd::EdgeEnd(Node* node, int index) noexcept : node_(node),
                                                         index_(index) {}

Node::EdgeEnd::EdgeEnd(std::shared_ptr<Node>& node) noexcept : EdgeEnd(node, INT_MAX) {}

Node::EdgeEnd::EdgeEnd(Node* node) noexcept : EdgeEnd(node, INT_MAX) {}

std::shared_ptr<Node> Graph::make_node() {
    MCHECK(nodes_.size() < static_cast<unsigned int>(std::numeric_limits<int>::max()));
    std::shared_ptr<Node> new_node(new Node(nodes_.size(), *this));
    nodes_.push_back(new_node);
    return new_node;
}

Node& Graph::add_node(const std::string& name, const std::string& op_type) {
    std::shared_ptr<Node> node = make_node();
    node->init(name, op_type);
    return *node;
}

void Scope::init(Graph *graph) {
    node_name_map.clear();
    for (auto it : graph->nodes()) {
        node_name_map.insert({it->name(), it});
    }
}

using NodeInputs = std::unordered_map<std::string, std::vector<std::string>>;
using NodeMap = std::unordered_map<std::string, std::shared_ptr<Node>>;

static void _sort_by_exe_order(const Graph *graph, NodeInputs& node_in,
                               std::vector<std::shared_ptr<Node>>& order_nodes,
                               std::set<std::string>& visited) {
    if (node_in.empty()) return;
    size_t node_size = graph->num_of_nodes();
    for (size_t i = 0; i < node_size; ++i) {
        if (visited.count(graph->nodes(i)->name()) == 1) continue;
        if (node_in.at(graph->nodes(i)->name()).size() == 0) { // input node
            order_nodes.push_back(graph->nodes(i));
            visited.insert(graph->nodes(i)->name());
            node_in.erase(graph->nodes(i)->name());
            for (auto& it : node_in) {
                for (auto iter = it.second.begin(); iter != it.second.end(); ++iter) {
                    if (*iter == graph->nodes(i)->name()) {
                        iter = it.second.erase(iter);
                        iter--;
                    }
                }
            }
        }
    }
    _sort_by_exe_order(graph, node_in, order_nodes, visited);
}

void Scope::sort_by_exe_order(Graph *graph) {
    NodeInputs node_inputs;
    NodeMap _node_name_map;
    for (auto it : graph->nodes()) {
        _node_name_map.insert({it->name(), it});
    }
    for (auto& it : graph->nodes()) {
        std::vector<std::string> inputs;
        for (auto& input : it->inputs()) {
            inputs.push_back(input->name());
        }
        node_inputs.insert({it->name(), inputs});
    }
    
    std::set<std::string> visited;
    std::vector<std::shared_ptr<Node>> order_nodes;
    _sort_by_exe_order(graph, node_inputs, order_nodes, visited);
    graph->nodes() = order_nodes;
}

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
    for (auto& node : graph.nodes()) {
        out<<"NodeName:"<<node->name()<<" index:"<<node->index()
           <<" OPType:"<<node->op_type()<<std::endl;
        for (auto& input : node->input_edges()) {
            Node* inode = input.get_node();
            out<<"    ---->InputName:"
               <<inode->name()<<" ctrl_index:"<<input.get_ctrl_index()
               <<" OPType:"<<inode->op_type()<<std::endl;
        }
        for (auto& output : node->output_edges()) {
            Node* onode = output.get_node();
            out<<"    ---->OutputName:"
               <<onode->name()<<" ctrl_index:"<<output.get_ctrl_index()
               <<" OPType:"<<onode->op_type()<<std::endl;
        }
    }
    
    return out;
}

} // namespace mariana
