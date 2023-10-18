/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : fold_reshape_to_node.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-05:17:01:59
 * Description:
 * 
 */

#include <maro/transform_utils.h>
#include <structure/ir.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>

namespace mariana { namespace transform {

/*
 *    ReshapeInput     AnyFusedReshape
 *     /  |  \            / | \
 *      Reshape  =======>   |
 *        |                 |
 *      AnyNode          AnyNode
 */
Status fold_reshape_to_node(Graph& graph) {
    auto func = [](const NodeMatch& match,
                   std::set<std::string>* old_nodes,
                   std::vector<std::shared_ptr<Node>>* new_nodes) -> Status {
        MVLOG(2)<<"Match:"<<match.debug_string();
        std::shared_ptr<Node> reshape_node = match.node;
        std::shared_ptr<Node> reshape_inode = match.inputs[0].node;
        std::vector<std::shared_ptr<Node>> onodes = onodes_of(reshape_node);

        for (auto& it : onodes) {
            for (auto& input : it->inputs()) {
                if (input == reshape_node->name()) {
                    input = reshape_inode->name();
                }
            }
        }
        reshape_inode->shapes() = reshape_node->shapes();
        old_nodes->insert(reshape_node->name());

        return absl::OkStatus();
    };
    replace_matching_optypes(graph,
                             {"FLATTEN",
                                 {
                                     {"*", {}}
                                 }
                             }, func);
    return absl::OkStatus();
}

Status base_fold_reshape_to_node(Graph& graph) {
    fold_reshape_to_node(graph);
    return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("base_fold_reshape_to_node", base_fold_reshape_to_node);

}} // namespace mariana::transform
