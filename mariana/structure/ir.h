/*
 *        (C) COPYRIGHT LeiNao Limited.
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
#include <core/utils/arrary_ref.h>
#include <core/macros/macros.h>
#include <core/impl/shape.h>

namespace mariana {

class Node;
class Graph;
using NodePtr = std::shared_ptr<Node>;
using ShapeList = std::vector<Shape>;
using NodeIndex = size_t;

class Node {
public:
    static bool enable_dynamic_shape() {
        return true;
    }

    Node() {}
    
    Node(NodeIndex index, Graph& graph) : index_(index), graph_(&graph), func_(nullptr) {}

    Node(const Node& rhs) {
        this->operator=(rhs);
    }

    Node& operator=(const Node& rhs) {
        if (this == &rhs) return *this;
        index_ = rhs.index_;
        graph_ = rhs.graph_;
        func_ = rhs.func_;
        op_ = rhs.op_;
        name_ = rhs.name_;
        relationships_ = rhs.relationships_;
        otensors_ = rhs.otensors_;
        return *this;
    }
    
    ~Node() {}
    
    NodeIndex index() const noexcept { return index_; }
    
    const std::string& name() const noexcept { return name_; }
    
    const std::string& op_type() const noexcept { return op_; }

    tensor_list apply(tensor_list&& inputs) {
        // for (relationships_.input_edges)
        return (*func_)(std::move(inputs));
    }

    void pre_run(ShapeList shapes) {
        for (auto it : shapes) {
            std::cout<<it<<std::endl;
        }
        oshapes_ = func_->infer_shape(shapes);
    }

    const ShapeList& shapes() const {
        return oshapes_;
    }

    ShapeList& shapes() {
        return oshapes_;
    }
    
    Function* op() noexcept { return func_.get(); }

    void init(const std::string& name, const std::string& op) {
        name_ = name;
        op_ = op;
        auto func_make = FunctionHolder::search(op);
        MCHECK(func_make!=nullptr)<<"There is no func in registry:"<<op;
        func_.reset(func_make());
    }
    
    class EdgeEnd {
    public:
        EdgeEnd(const Node* node, int index) noexcept;
        explicit EdgeEnd(const Node* node) noexcept;
        const Node& get_node() const noexcept { return *node_; }
        int get_index() const { return index_; }
    private:
        const Node* node_ = nullptr;
        const int index_ = INT_MAX;
    };

    struct EdgeEndCompare {
        bool operator()(const EdgeEnd& lhs, const EdgeEnd& rhs) const {
            if (lhs.get_node().index() == rhs.get_node().index()) {
                return lhs.get_index() < rhs.get_index();
            }
            return lhs.get_node().index() < rhs.get_node().index();
        }
    };
    using EdgeSet = std::set<EdgeEnd, EdgeEndCompare>;

    class Relationships {
    public:
        Relationships() = default;
        ~Relationships() {
            clear();
        }
        Relationships(const Relationships& rhs) {
            this->operator=(rhs);
        }
        Relationships& operator=(const Relationships& rhs) {
            if (this == &rhs) return *this;
            input_edges = rhs.input_edges;
            output_edges = rhs.output_edges;
            control_inputs = rhs.control_inputs;
            return *this;
        }
        void clear() noexcept {
            input_edges.clear();
            output_edges.clear();
            control_inputs.clear();
        }
        
        size_t isize() const {
            return input_edges.size();
        }

        size_t osize() const {
            return output_edges.size();
        }
        
        /** The edges for Nodes that provide inputs to this Node. */
        EdgeSet input_edges;
        /** The edges for Nodes that receive outputs from this Node. */
        EdgeSet output_edges;
        /** The Node names of the control inputs to this Node. */
        std::set<std::string> control_inputs;
        
    };
    Relationships& relationships() {
        return relationships_;
    }
    const Relationships& relationships() const {
        return relationships_;
    }
    friend class Graph;
private:
    NodeIndex index_ = std::numeric_limits<NodeIndex>::max();
    Graph* graph_ = nullptr;
    std::shared_ptr<Function> func_;
    std::string op_ = "";
    std::string name_ = "";
    Relationships relationships_;
    tensor_list otensors_;
    ShapeList oshapes_;
};

class Graph {
public:
    Graph() {}
    Graph(const Graph& rhs) {
        this->operator=(rhs);
    }
    Graph& operator=(const Graph& rhs) {
        if (this == &rhs) {
            return *this;
        }
        nodes_ = rhs.nodes_;
        name_ = rhs.name_;
        num_of_nodes_ = rhs.num_of_nodes_;
        return *this;
    }
    Node& add_node(const std::string& name, const std::string& op_type);
    Node* make_node();
    size_t num_of_nodes(void) const {
        return num_of_nodes_;
    }
    const std::vector<std::shared_ptr<Node>>& nodes() const {
        return nodes_;
    }
    std::vector<std::shared_ptr<Node>>& nodes() {
        return nodes_;
    }
    const std::shared_ptr<Node>& nodes(size_t i) const {
        return nodes_[i];
    }
    std::shared_ptr<Node>& nodes(size_t i) {
        return nodes_[i];
    }
private:
    std::vector<std::shared_ptr<Node>> nodes_;
    std::string name_ = "";
    size_t num_of_nodes_ = 0;
};

struct Scope {
    Scope(Graph *graph) {
        init(graph);
    }
    ~Scope() {}
    void init(Graph *graph);
    std::unordered_map<std::string, std::shared_ptr<Node>> node_name_map;
};

} // namespace mariana

#endif /* __STRUCTURE_IR_H__ */

