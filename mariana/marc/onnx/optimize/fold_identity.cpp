/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/optimize/fold_identity.cpp
 * Authors    : lqwang@pandora
 * Create Time: 2022-12-30:21:14:20
 * Description:
 * 
 */

#include <marc/onnx/ops.h>
#include <marc/onnx/optimize/transform_utils.h>

namespace mariana { namespace onnx { namespace transform {

Status fold_identity_to_conv(OnnxScope& scope) {
    ::onnx::GraphProto replaced_graph;
    auto func = [](OnnxScope& scope, const NodeMatch& match,
       std::set<std::string>* old_nodes,
       std::set<std::string>* old_tensors,
       std::vector<::onnx::NodeProto>* new_nodes,
       std::vector<::onnx::TensorProto>* new_tensors) -> Status {
        const ::onnx::NodeProto& conv_node = match.node;
        const ::onnx::NodeProto& identity_node = match.inputs[1].node;
        ::onnx::NodeProto new_node;
        new_node.CopyFrom(conv_node);
        new_node.set_input(2, scope.nodes_info[identity_node.name()].tensors[0]->name());
        new_nodes->push_back(new_node);
        old_nodes->insert(conv_node.name());
        old_nodes->insert(identity_node.name());
        return absl::OkStatus();
    };
    
    bool job;
    do {
        replace_matching_optypes(scope,
                                 {"Conv",
                                     {
                                         {"*"},
                                         {"Identity"}
                                     }
                                 }, func, &replaced_graph);
        scope.update(replaced_graph);
        job = false;
        for (auto& node : replaced_graph.node()) {
            if (node.op_type() == KIdentity) {
                job = true;
                break;
            }
        }
    } while (job);
    scope.save("fold_identity_to_conv.onnx");
    return absl::OkStatus();
}

REGISTER_ONNX_GRAPH_TRANSFORM("fold_identity_to_conv", fold_identity_to_conv);

}}} // namespace mariana::onnx::transform
