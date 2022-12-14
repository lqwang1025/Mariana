/*
 *        (C) COPYRIGHT LeiNao Limited.
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

namespace mariana { namespace transform {

Status base_fold_reshape_to_node(Graph& graph) {
    Graph replaced_graph;
    Scope scope(&graph);
    auto func = [](Scope& scope, const NodeMatch& match,
                   std::set<std::string>* old_nodes,
                   std::vector<Node>* new_nodes) -> Status {
        std::cout<<"deu:"<<match.debug_string()<<std::endl;
        const Node& any_node = match.node;
        const Node& reshape_node = match.inputs[0].node;
        const Node& reshape_inode = match.inputs[0].inputs[0].node;
        
        Node new_node = any_node;
        new_node.clear_input();
        for(size_t i = 0; i < reshape_node.inputs().size(); ++i) {
            new_node.update_input(reshape_node.inputs()[i], i);
        }
        
        Node new_inode = reshape_inode;
        new_inode.clear_output();
        new_inode.shapes().clear();
        new_inode.shapes().insert(new_inode.shapes().begin(),
                                  reshape_node.shapes().begin(),
                                  reshape_node.shapes().end());
        for(size_t i = 0; i < reshape_node.outputs().size(); ++i) {
            new_inode.update_output(reshape_node.outputs()[i], i);
        }
        
        new_nodes->push_back(new_node);
        new_nodes->push_back(new_inode);
        old_nodes->insert(reshape_node.name());
        old_nodes->insert(any_node.name());
        old_nodes->insert(reshape_inode.name());
        return absl::OkStatus();
    };
    replace_matching_optypes(graph,
                             {"*",
                                 {
                                     {"Reshape", {{"*"}}}
                                 }
                             }, func, &replaced_graph);
    graph = replaced_graph;
    return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("base_fold_reshape_to_node", base_fold_reshape_to_node);

}} // namespace mariana::transform
