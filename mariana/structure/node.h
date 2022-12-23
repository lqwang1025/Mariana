/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/node.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:12:09
 * Description:
 *
 */

#ifndef __STRUCTURE_NODE_H__
#define __STRUCTURE_NODE_H__

#include <vector>
#include <structure/edge.h>
#include <structure/tensor.h>

namespace mariana {

using tensor_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

struct Node {
    explicit Node(uint64_t sequence_nr, edge_list&& next_edges = edge_list())
        : sequence_nr_(sequence_nr), next_edges_(std::move(next_edges)) {
        for (const Edge& edge : next_edges_) {
            update_topological_nr(edge);
        }

        if (AnomalyMode::is_enabled()) {
            metadata()->store_stack();

            // If anomaly mode is enabled and graph is constructed, then assign the
            // currently evaluating node as the parent of this node.
            // A parent is a Node where this Node is created.
            // We are tracking the parents to track multiple backward operations.
            assign_parent();
        }

        // Store the thread_id of the forward operator.
        // See NOTE [ Sequence Numbers ]
        thread_id_ = at::RecordFunction::currentThreadId();
    }
    explicit Node(edge_list&& next_edges = edge_list())
        : Node(
            /*sequence_nr=*/at::sequence_number::get_and_increment(),
            std::move(next_edges)) {}
    Node(const Node& other) = delete;
    Node(Node&& other) = delete;
    Node& operator=(const Node& other) = delete;
    Node& operator=(Node&& other) = delete;
    virtual ~Node() = default;
};

} // namespace mariana

#endif /* __STRUCTURE_NODE_H__ */

