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
                              },
                              [](const NodeMatch& match, 
                                 std::vector<::onnx::NodeProto>* new_nodes) -> Status {
                                  std::cout<<"dd:"<<match.debug_string()<<std::endl;
                              },
                                  
                              }, &replaced_graph);
}

REGISTER_ONNX_GRAPH_TRANSFORM("fold_identity_to_conv", fold_identity_to_conv);

}}} // namespace mariana::onnx::transform
