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
    context.otensors.clear();
    if (graph.engine_) {
        graph.engine_->run(context);
        for (size_t i = 0; i < graph.engine_->otensors.size(); ++i) {
            MTensor tensor;
            Tensor& src = graph.engine_->otensors[i];
            for (auto it : src.shape().data()) {
                tensor.shape.push_back(it);
            }
            tensor.input = src.data();
            tensor.dtype = src.dtype();
            tensor.device = src.device().type();
            context.otensors.push_back(tensor);
        }
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

void GraphExec::pre_run(Graph& graph, const ConvertContext& context) {
    for (auto& node : graph.order()) {
        auto& inputs = node->inputs();
        if (inputs.size() == 0) {
            node->pre_run({Shape{context.ishapes.at(node->name())}});
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
