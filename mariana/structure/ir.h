/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/ir.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-28:08:55:17
 * Description:
 *
 */

#ifndef __STRUCTURE_IR_H__
#define __STRUCTURE_IR_H__

#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

#include <structure/function.h>
#include <structure/processor.h>
#include <structure/engine.h>
#include <core/utils/arrary_ref.h>
#include <core/macros/macros.h>
#include <core/impl/shape.h>

namespace mariana {

class Node;
class Graph;
using ShapeList = std::vector<Shape>;
using NodeList = std::vector<Node*>;

class Node {
public:
    static bool enable_dynamic_shape() {
        return true;
    }

    Node() {}
    
    Node(Graph& graph) : graph_(&graph), func_(nullptr) {
        oshapes_.clear();
        otensors_.clear();
        inames_.clear();
    }

    Node(const Node& rhs) {
        this->operator=(rhs);
    }

    Node& operator=(const Node& rhs) {
        if (this == &rhs) return *this;
        graph_    = rhs.graph_;
        func_     = rhs.func_;
        op_       = rhs.op_;
        name_     = rhs.name_;
        inames_   = rhs.inames_;
        otensors_ = rhs.otensors_;
        oshapes_  = rhs.oshapes_;
        ctrl_idx_ = rhs.ctrl_idx_;
        return *this;
    }
    
    ~Node() {
        oshapes_.clear();
        otensors_.clear();
        inames_.clear();
    }
    
    std::vector<std::string> inputs() const {
        return inames_;
    }

    std::vector<std::string>& inputs() {
        return inames_;
    }
    
    std::vector<int32_t> ctrl_idx() const {
        return ctrl_idx_;
    }

    std::vector<int32_t>& ctrl_idx() {
        return ctrl_idx_;
    }

    Graph* graph() const {
        return graph_;
    }
    
    const std::string& name() const noexcept { return name_; }
    
    const std::string& op_type() const noexcept { return op_; }

    tensor_list apply(tensor_list&& inputs) {
        return (*func_)(std::move(inputs));
    }
 
    void pre_run(ShapeList shapes) {
        oshapes_ = func_->infer_shape(shapes);
        /* for (auto& it : oshapes_) { */
        /*     std::cout<<"Infer shape:"<<name()<<" "<<it<<std::endl; */
        /* } */
        float flops = func_->compute_FLOPs(oshapes_);
        // std::cout<<"debug:"<<name()<<" "<<flops<<std::endl;
    }
    
    const ShapeList& shapes() const {
        return oshapes_;
    }

    ShapeList& shapes() {
        return oshapes_;
    }
    
    Function* op() const noexcept { return func_.get(); }

    void init(const std::string& name, const std::string& op) {
        name_ = name;
        op_ = op;
        auto func_make = FunctionHolder::search(op);
        MCHECK(func_make!=nullptr)<<"There is no func in registry:"<<op;
        func_.reset(func_make());
    }
private:
    friend class Graph;
    friend std::ostream& operator<<(std::ostream& out, const Node& node);
    Graph* graph_ = nullptr;
    std::shared_ptr<Function> func_;
    std::string op_ = "";
    std::string name_ = "";
    std::vector<std::string> inames_;
    std::vector<int32_t> ctrl_idx_;
    tensor_list otensors_;
    ShapeList oshapes_; // dep
};

std::ostream& operator<<(std::ostream& out, const Node& node);

class Graph {
public:
    Graph() {
        node_map_.clear();
    }
    
    Graph(std::shared_ptr<Engine> engine) : engine_(engine) {
        Graph();
    }

    Graph(const Graph& rhs) {
        this->operator=(rhs);
    }
    
    Graph& operator=(const Graph& rhs) {
        if (&rhs == this) return *this;
        name_      = rhs.name_;
        engine_    = rhs.engine_;
        processor_ = rhs.processor_;
        node_map_  = rhs.node_map_;
        nodes_     = rhs.nodes_;
        return *this;
    }
    
    ~Graph() {
        node_map_.clear();
    }
    
    Graph& finilize();
    
    Node& new_node(const std::string& name, const std::string& op_type);

    std::shared_ptr<Engine> engine() const {
        return engine_;
    }
    
    size_t num_of_nodes(void) const {
        return node_map_.size();
    }
    
    std::vector<std::shared_ptr<Node>> order() const {
        return nodes_;
    }

    Graph& update_node(const std::string& name, std::shared_ptr<Node> node) {
        node_map_[name] = node;
        return *this;
    }

    Graph& update_node(std::shared_ptr<Node> node) { // update node iteself.
        node_map_[node->name()] = node;
        return *this;
    }

    Graph& remove_node(const std::string& name) {
        node_map_.erase(name);
        return *this;
    }

    Graph& remove_node(std::shared_ptr<Node> node) {
        node_map_.erase(node->name());
        return *this;
    }
    
    std::shared_ptr<Node> node(const std::string& name) const {
        if (node_map_.count(name)) {
            return node_map_.at(name);
        } else {
            return nullptr;
        }
    }
    std::shared_ptr<Node> node(const std::string& name) {
        if (node_map_.count(name)) {
            return node_map_[name];
        } else {
            return nullptr;
        }
    }
    void set_pro(Processor* processor) {
        processor_.reset(processor);
    }
private:
    friend class GraphExec;
    friend std::ostream& operator<<(std::ostream& out, const Graph& graph);
    std::string name_ = "";
    std::shared_ptr<Engine> engine_ = nullptr;
    std::shared_ptr<Processor> processor_ = nullptr;
    std::unordered_map<std::string, std::shared_ptr<Node>> node_map_;
    std::vector<std::shared_ptr<Node>> nodes_;
};

std::ostream& operator<<(std::ostream& out, const Graph& graph);

std::vector<std::shared_ptr<Node>> inodes_of(std::shared_ptr<Node>& node);
std::vector<std::shared_ptr<Node>> onodes_of(std::shared_ptr<Node>& node);

} // namespace mariana

#endif /* __STRUCTURE_IR_H__ */

