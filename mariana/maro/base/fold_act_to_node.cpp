/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : fold_act_to_node.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:14:04:49
 * Description:
 * 
 */

#include <maro/transform_utils.h>
#include <structure/ir.h>
#include <structure/funcs/ops.h>

namespace mariana { namespace transform {

Status base_fold_act_to_node(Graph& graph) {
    auto func = [](const NodeMatch& match,
                   std::set<std::string>* old_nodes,
                   std::vector<std::shared_ptr<Node>>* new_nodes) -> Status {
        // const Node& relu_node = match.node;
        // const Node& input_node = match.inputs[0].node;
        // ::onnx::NodeProto new_node;
        // new_nodes->push_back(new_node);
        // old_nodes->insert(relu_node.name());
        return absl::OkStatus();
    };
    replace_matching_optypes(graph,
                             {"RELU",
                                 {
                                     {"*"}
                                 }
                             }, func);
    return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("base_fold_act_to_node", base_fold_act_to_node);

}} // namespace mariana::transform
