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

#include <marc/onnx/optimize/transform_utils.h>

namespace mariana { namespace onnx { namespace transform {

Status fold_identity_to_conv(OnnxScope& scope) {
    ::onnx::GraphProto replaced_graph;
    replace_matching_optypes(scope,
                             {"Conv",
                              {
                                  {"*"},
                                  {"Identity"}
                              }
                             },
                             [](OnnxScope& scope, const NodeMatch& match, 
                                std::vector<::onnx::NodeProto>* new_nodes,
                                std::vector<::onnx::TensorProto>* new_tensors) -> Status {
                                 const ::onnx::NodeProto& conv_node = match.node;
                                 const ::onnx::NodeProto& i0_node = match.inputs[0].node;
                                 const ::onnx::NodeProto& identity_node = match.inputs[1].node;
                                 ::onnx::TensorProto new_tensor;
                                 new_tensor.CopyFrom(*scope.nodes_info[identity_node.name()].tensors[0]);
                                 ::onnx::NodeProto new_node;
                                 new_node.CopyFrom(conv_node);
                                 new_node.set_input(2, new_tensor.name());
                                 new_nodes->push_back(new_node);
                                 new_nodes->push_back(i0_node);
                                 new_tensors->push_back(new_tensor);
                                 return absl::OkStatus();
                             }, &replaced_graph);
    return absl::OkStatus();
}

REGISTER_ONNX_GRAPH_TRANSFORM("fold_identity_to_conv", fold_identity_to_conv);

}}} // namespace mariana::onnx::transform
