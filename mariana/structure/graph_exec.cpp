/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/graph_exec.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:16:21:37
 * Description:
 * 
 */

#include <structure/graph_exec.h>
#include <iostream>

namespace mariana {

void GraphExec::run(Graph& graph, ExecContext& context) {
    
}

void GraphExec::pre_run(Graph& graph, ExecContext& context) {
    for (auto& node : graph.nodes()) {
        auto& relationships = node->relationships();
        if (relationships.isize() == 0) {
            node->pre_run({context.ishapes.at(node->name())});
            std::cout<<"debug:"<<node->name()<<std::endl;
        } else {
            for (auto it : relationships.input_edges) {
                node->pre_run(it.get_node().shapes());
            }
        }
    }
}

} // namespace mariana
