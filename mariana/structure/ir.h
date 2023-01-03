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
using OpList = ArrayRef<NodePtr>;
using NodeIndex = size_t;

class Node {
public:
    static bool enable_dynamic_shape() {
        return true;
    }
    
    Node(NodeIndex index, Graph& graph) : index_(index), graph_(&graph), func_(nullptr) {}
    
    ~Node() { delete func_; }
    
    NodeIndex index() const noexcept { return index_; }
    
    const std::string& name() const noexcept { return name_; }
    
    const std::string& op_type() const noexcept { return op_; }

    Function* op() noexcept { return func_; }

    void init(const std::string& name, const std::string& op) {
        name_ = name;
        op_ = op;
        auto func_make = FunctionHolder::search(op);
        MCHECK(func_make!=nullptr)<<"There is no func in registry:"<<op;
        func_ = func_make();
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
        void clear() noexcept {
            input_edges.clear();
            output_edges.clear();
            control_inputs.clear();
        }
        /** The edges for Nodes that provide inputs to this Node. */
        EdgeSet input_edges;
        /** The edges for Nodes that receive outputs from this Node. */
        EdgeSet output_edges;
        /** The Node names of the control inputs to this Node. */
        std::set<std::string> control_inputs;
    private:
        MAR_DISABLE_COPY_AND_ASSIGN(Relationships);
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
    Graph* graph_;
    Function* func_;
    std::string op_;
    std::string name_;
    Relationships relationships_;
    
};

class Graph {
public:
    Graph() {}
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

