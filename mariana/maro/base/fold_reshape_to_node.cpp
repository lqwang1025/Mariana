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
 *         |                |
 *      Reshape  =======>   |
 *         |                |
 *      AnyNode          AnyNode
 */

Status base_fold_reshape_to_node(Graph& graph) {
    Graph replaced_graph;
    Scope scope(&graph);
    auto func = [](Scope& scope, const NodeMatch& match,
                   std::set<std::string>* old_nodes,
                   std::vector<std::shared_ptr<Node>>* new_nodes) -> Status {
        MVLOG(2)<<"Match:"<<match.debug_string();
        const Node& any_node = match.node;
        const Node& reshape_node = match.inputs[0].node;
        const Node& reshape_inode = match.inputs[0].inputs[0].node;
        
        std::shared_ptr<Node> new_node = std::make_shared<Node>();
        *new_node = any_node;

        /*
         *  build link for:  AnyNode  
         *                    / | \
         */
        for (auto& output : new_node->outputs()) {
            for (size_t i = 0; i < output->inputs().size(); ++i) {
                if (output->inputs()[i]->name() == new_node->name()) {
                    output->update_input(new_node.get(), i);
                    break;
                }
            }
        }
        
        std::shared_ptr<Node> new_inode = std::make_shared<Node>();
        *new_inode = reshape_inode;
        new_inode->clear_output();
        new_inode->shapes().clear();
        new_inode->shapes().insert(new_inode->shapes().begin(),
                                   reshape_node.shapes().begin(),
                                   reshape_node.shapes().end());

        /*                      
         *  build link for:  new_inode
         *                      |
         */
        new_inode->update_output(new_node.get(), 0);

        /*                    \ | /
         *  build link for:  new_inode  
         *                    
         */
        for (auto& input : new_inode->inputs()) {
            for (size_t i = 0; i < input->outputs().size(); ++i) {
                if (input->outputs()[i]->name() == new_inode->name()) {
                    input->update_output(new_inode.get(), i);
                    break;
                }
            }
        }

        /*                      |
         *  build link for:  AnyNode
         */
        new_node->clear_input();
        new_node->update_input(new_inode.get(), 0);
        
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
                                     {MRESHAPE, {{"*"}}}
                                 }
                             }, func, &replaced_graph);
    graph = replaced_graph;
    return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("base_fold_reshape_to_node", base_fold_reshape_to_node);

}} // namespace mariana::transform
