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

#include <structure/ir.h>
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
        } else {
            ShapeList shapes;
            for (auto it : relationships.input_edges) {
                shapes.insert(shapes.end(), it.get_node().shapes().begin(),
                              it.get_node().shapes().end());
            }
            node->pre_run(shapes);
        }
    }
}

} // namespace mariana
