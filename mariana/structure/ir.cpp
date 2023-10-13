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

std::vector<std::shared_ptr<Node>> inodes_of(std::shared_ptr<Node>& node) {
    std::vector<std::shared_ptr<Node>> nodes;
    for (auto& it : node->inputs()) {
        nodes.push_back(node->graph()->node(it));
    }
    return nodes;
}

std::vector<std::shared_ptr<Node>> onodes_of(std::shared_ptr<Node>& node) {
    std::vector<std::shared_ptr<Node>> nodes;
    for (auto& it : node->graph()->order()) {
        for (auto& input : it->inputs()) {
            if (input == node->name()) {
                nodes.push_back(it);
            }
        }
    }
    return nodes;
}

Node& Graph::new_node(const std::string& name, const std::string& op_type) {
    MCHECK(node_map_.size() < static_cast<unsigned int>(std::numeric_limits<int>::max()))<<"Grpah ";
    std::shared_ptr<Node> node(new Node(*this));
    node->init(name, op_type);
    node_map_[name] = node;
    return *node;
}

using NodeInputs = std::unordered_map<std::string, std::vector<std::string>>;
void _sort_by_exe_order(std::unordered_map<std::string,
                        std::shared_ptr<Node>>& node_map, NodeInputs& node_in,
                        std::vector<std::shared_ptr<Node>>& order_nodes,
                               std::set<std::string>& visited) {
    if (node_in.empty()) return;
    for (auto& it : node_map) {
        if (visited.count(it.first) == 1) continue;
        if (node_in.at(it.first).size() == 0) { // input node
            order_nodes.push_back(it.second);
            visited.insert(it.first);
            node_in.erase(it.first);
            for (auto& itt : node_in) {
                   for (auto iter = itt.second.begin(); iter != itt.second.end(); ++iter) {
                       if (*iter == it.first) {
                           iter = itt.second.erase(iter);
                           iter--;
                       }
                   }
            }
        }
    }
    _sort_by_exe_order(node_map, node_in, order_nodes, visited);
}

Graph& Graph::finilize() {
    NodeInputs node_inputs;
    for (auto& it : node_map_) {
        node_inputs.insert({it.first, it.second->inputs()});
    }
    std::set<std::string> visited;
    std::unordered_map<std::string, std::shared_ptr<Node>> node_map = node_map_;
    nodes_.clear();
    _sort_by_exe_order(node_map_, node_inputs, nodes_, visited);
    return *this;
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
    out<<"Node name:["<<node.name_<<"], OpType:["<<node.op_
       <<"], Graph ptr:["<<node.graph_<<"], Op ptr:["<<node.func_
       <<"]"<<std::endl;
    out<<"--->Input names:[";
    for (size_t i = 0; i < node.inames_.size(); ++i) {
        out<<node.inames_[i];
        if (i != node.inames_.size()-1) {
            out<<", ";
        }
    }
    
    out<<"]\n--->Oshapes:";
    for (auto& it : node.oshapes_) {
        out<<it;
    }
    out<<std::endl;
    return out;
}

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
    out<<"Graph node size:"<<graph.num_of_nodes()<<" engine:"<<graph.engine_
       <<" processor:"<<graph.processor_<<" name:"<<graph.name_<<std::endl;
    std::vector<std::shared_ptr<Node>> orders = graph.order();
    for (auto& node : orders) {
        out<<*node;
    }
    return out;
}

} // namespace mariana
