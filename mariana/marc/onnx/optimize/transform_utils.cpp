/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/optimize/transform_utils.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:16:03:04
 * Description:
 * 
 */

#include <absl/strings/str_split.h>
#include <marc/onnx/onnx.h>
#include <marc/onnx/optimize/transform_utils.h>

namespace mariana { namespace onnx { namespace transform {

GraphMatcher::GraphMatcher(OnnxScope& scope) : scope_(scope) {
    OnnxScope::sort_by_execution_order(*scope.graph_info.graph, &graph_);
}

bool _does_optype_match(const ::onnx::NodeProto& node,
                        const OpTypePattern& pattern,
                        const std::set<std::string>& previously_matched_nodes,
                        NodeMatch* match) {
    if (previously_matched_nodes.count(node.name())) {
        MLOG(INFO) << "node " << node.name() << " has been previously matched";
        return false;
    }
      bool pattern_matched = false;
      if (pattern.op == "*") {
          pattern_matched = true;
      } else {
          std::vector<std::string> pattern_ops = absl::StrSplit(pattern.op, '|');
          for (const std::string& pattern_op : pattern_ops) {
              if (node.op() == pattern_op) {
                  pattern_matched = true;
              }
          }
      }
      if (!pattern_matched) {
          MLOG(INFO) << "node.op() != pattern.op()";
          return false;
      }
      match->node = node;
      // Ignore any control inputs for pattern-matching purposes
      std::vector<std::string> non_control_inputs;
      for (const std::string& input : node.input()) {
          if (!input.empty() && (input[0] != '^')) {
              non_control_inputs.push_back(input);
          }
      }
      if (pattern.inputs.empty()) {
          // If there are no inputs, assume that's the end of the pattern.
          return true;
      }

      if (non_control_inputs.size() != pattern.inputs.size()) {
          MLOG(INFO) << "non_control_inputs.size() != pattern.inputs.size()";
          return false;
      }
      for (int i = 0; i < pattern.inputs.size(); ++i) {
          const string& input_node_name = non_control_inputs[i];
          const NodeDef& input_node = *(node_map_[input_node_name]);
          const OpTypePattern& input_pattern = pattern.inputs[i];
          match->inputs.push_back(NodeMatch());
          NodeMatch* input_match = &(match->inputs.back());
          if (!DoesOpTypeMatch(input_node, input_pattern, previously_matched_nodes,
                               input_match)) {
              return false;
          }
      }
      return true;
}

}}} // namespace mariana::onnx::transform
