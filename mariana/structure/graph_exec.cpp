/*
 *        (C) COPYRIGHT Daniel Limited.
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
    if (graph.engine_) {
        graph.engine_->run(context);
    }
    if (graph.processor_) {
        results = (*graph.processor_)(std::move(graph.engine_->otensors), context);
    }
}

void GraphExec::pre_run(Graph& graph, ExecContext& context) {
    for (auto& node : graph.order()) {
        auto& inputs = node->inputs();
        if (inputs.size() == 0) {
            node->pre_run({Shape{context.itensors.at(node->name()).shape}});
        } else {
            ShapeList shapes;
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::shared_ptr<Node> inode = graph.node(inputs.at(i));
                int32_t ctrl_idx = node->ctrl_idx()[i];
                shapes.push_back(inode->shapes()[ctrl_idx]);
            }
            node->pre_run(shapes);
        }
    }
}

void GraphExec::pre_run(Graph& graph, const proto::ModelInfo& model_info) {
    for (auto& node : graph.order()) {
        auto& inputs = node->inputs();
        if (inputs.size() == 0) {
            std::vector<int32_t> shape;
            shape.reserve(model_info.ishapes().at(node->name()).dim_size());
            for (size_t i = 0; i < model_info.ishapes().at(node->name()).dim_size(); ++i) {
                shape.push_back(model_info.ishapes().at(node->name()).dim(i));
            }
            node->pre_run({Shape{shape}});
        } else {
            ShapeList shapes;
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::shared_ptr<Node> inode = graph.node(inputs.at(i));
                int32_t ctrl_idx = node->ctrl_idx()[i];
                shapes.push_back(inode->shapes()[ctrl_idx]);
            }
            node->pre_run(shapes);
        }
    }
}


} // namespace mariana
