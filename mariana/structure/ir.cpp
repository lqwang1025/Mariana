/*
 *        (C) COPYRIGHT LeiNao Limited.
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

Node::EdgeEnd::EdgeEnd(const Node& node, int src_arg_index,
                 int dst_arg_index) noexcept : node_(&node),
    src_arg_index_(src_arg_index),
    dst_arg_index_(dst_arg_index) {}

Node::EdgeEnd::EdgeEnd(const Node& node) noexcept : EdgeEnd(node, INT_MAX, INT_MAX) {}

Node::NodeConstIterator::NodeConstIterator(EdgeConstIterator p_iter) {
    m_iter = p_iter;
}

bool Node::NodeConstIterator::operator==(const NodeConstIterator& p_other) const {
    return m_iter == p_other.m_iter;
}

bool Node::NodeConstIterator::operator!=(const NodeConstIterator& p_other) const {
    return m_iter != p_other.m_iter;
}

void Node::NodeConstIterator::operator++() {
    ++m_iter;
}

void Node::NodeConstIterator::operator--() {
    --m_iter;
}

const Node& Node::NodeConstIterator::operator*() const {
    return (*m_iter).get_node();
}

const Node* Node::NodeConstIterator::operator->() const {
    return &(operator*());
}

Node* Graph::make_node() {
    MCHECK(nodes_.size() < static_cast<unsigned int>(std::numeric_limits<int>::max()));
    std::unique_ptr<Node> new_node(new Node(nodes_.size(), *this));
    Node* node{new_node.get()};
    nodes_.push_back(std::move(new_node));
    ++num_of_nodes_;
    return node;
}

Node& Graph::add_node(const std::string& name, const std::string& op_type) {
    Node* node = make_node();
    node->init(name, op_type);
    return *node;
}

} // namespace mariana
