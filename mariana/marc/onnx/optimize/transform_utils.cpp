/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/optimize/transform_utils.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:16:03:04
 * Description:
 * 
 */

#include <set>

#include <absl/strings/str_split.h>
#include <marc/onnx/onnx.h>
#include <marc/onnx/optimize/transform_utils.h>

namespace mariana { namespace onnx { namespace transform {

std::string OpTypePattern::debug_string() const {
    std::string result = "{" + op + ", {";
    for (const OpTypePattern& input : inputs) {
        result += input.debug_string() + ",";
    }
    result += "}}";
    return result;
}

std::string NodeMatch::debug_string() const {
    std::string result = "{";
    result += node.name() + ":" + node.op_type();
    result += ", {";
    for (const NodeMatch& input : inputs) {
        result += input.debug_string() + ",";
    }
    result += "}}";
    return result;
}

GraphMatcher::GraphMatcher(OnnxScope& scope) : scope_(scope) {
    OnnxScope::sort_by_execution_order(*scope.graph_info.graph, &graph_);
}

static void record_matched_nodes(const NodeMatch& match,
                                 std::set<std::string>* matched_nodes) {
    matched_nodes->insert(match.node.name());
    for (const NodeMatch& input_match : match.inputs) {
        record_matched_nodes(input_match, matched_nodes);
    }
}

Status GraphMatcher::get_optype_matches(const OpTypePattern& pattern,
                                        std::vector<NodeMatch>* matches) {
    std::set<std::string> matched_nodes;
    for (const ::onnx::NodeProto& node : graph_.node()) {
        // Skip any nodes that are already part of a match.
        if (matched_nodes.count(node.name())) {
            continue;
        }
        NodeMatch match;
        if (_does_optype_match(node, pattern, matched_nodes, &match)) {
            record_matched_nodes(match, &matched_nodes);
            matches->push_back(match);
        }
    }
    return absl::OkStatus();
}

bool GraphMatcher::_does_optype_match(const ::onnx::NodeProto& node,
                                      const OpTypePattern& pattern,
                                      const std::set<std::string>& previously_matched_nodes,
                                      NodeMatch* match) {
    MVLOG(1) << "Looking at node " << node.name()<<" "<<node.op_type();
    MVLOG(1) << "pattern=" << pattern.debug_string();
    MVLOG(1) << "match=" << match->debug_string();

    if (previously_matched_nodes.count(node.name())) {
        MVLOG(1) << "node " << node.name() << " has been previously matched";
        return false;
    }
    bool pattern_matched = false;
    if (pattern.op == "*") {
        pattern_matched = true;
    } else {
        std::vector<std::string> pattern_ops = absl::StrSplit(pattern.op, '|');
        for (const std::string& pattern_op : pattern_ops) {
            if (node.op_type() == pattern_op) {
                pattern_matched = true;
            }
        }
    }
    if (!pattern_matched) {
        MVLOG(1) << "node.op() != pattern.op"
                 << node.op_type()<<" "<<pattern.op;
        return false;
    }
    match->node = node;
    if (pattern.inputs.empty()) {
        // If there are no inputs, assume that's the end of the pattern.
        return true;
    }
      
    // Ignore any control inputs for pattern-matching purposes
    std::vector<std::string> non_control_inputs;
    for (::onnx::NodeProto* input : scope_.nodes_info[node.name()].nodes) {
        if (!input->name().empty()) {
            non_control_inputs.push_back(input->name());
        }
    }

    if (non_control_inputs.size() != pattern.inputs.size()) {
        MVLOG(1) << "non_control_inputs.size() != pattern.inputs.size() "
                 << non_control_inputs.size()<<" : "<<pattern.inputs.size();
        return false;
    }
    for (size_t i = 0; i < pattern.inputs.size(); ++i) {
        const std::string& input_node_name = non_control_inputs[i];
        const ::onnx::NodeProto& input_node = *(scope_.graph_info.node_name_map[input_node_name]);
        const OpTypePattern& input_pattern = pattern.inputs[i];
        match->inputs.push_back(NodeMatch());
        NodeMatch* input_match = &(match->inputs.back());
        if (!_does_optype_match(input_node, input_pattern, previously_matched_nodes,
                                input_match)) {
            return false;
        }
    }
    return true;
}

TransformRegistry* get_transform_registry() {
    static TransformRegistry transform_registry;
    return &transform_registry;
}

Status replace_matching_optypes(OnnxScope& scope,
                                const OpTypePattern& pattern,
                                const std::function<
                                Status(OnnxScope& scope, const NodeMatch&,
                                       std::set<std::string>*, std::set<std::string>*,
                                       std::vector<::onnx::NodeProto>*,
                                       std::vector<::onnx::TensorProto>*)>& node_generator,
                                ::onnx::GraphProto* graph) {
    GraphMatcher matcher{scope};
    std::vector<NodeMatch> matches;
    Status res = matcher.get_optype_matches(pattern, &matches);
    if (!res.ok()) {
        MLOG(ERROR) << "get_optype_matches error";
        return res;
    }
    Status status = absl::OkStatus();
    graph->CopyFrom(*scope.graph_info.graph);
    std::set<std::string> old_nodes;
    std::set<std::string> old_tensors;
    std::vector<::onnx::NodeProto> new_nodes;
    std::vector<::onnx::TensorProto> new_tensors;
    for (const NodeMatch& match : matches) {
        std::vector<::onnx::NodeProto> _new_nodes;
        std::vector<::onnx::TensorProto> _new_tensors;
        status = node_generator(scope, match, &old_nodes, &old_tensors, &_new_nodes, &_new_tensors);
        new_nodes.insert(new_nodes.end(), _new_nodes.begin(), _new_nodes.end());
        new_tensors.insert(new_tensors.end(), _new_tensors.begin(), _new_tensors.end());
    }

    std::vector<::onnx::NodeProto> reserve_nodes;
    reserve_nodes.reserve(graph->node_size());
    
    for (auto& node : graph->node()) {
        if (old_nodes.count(node.name())) {
            continue;
        }
        reserve_nodes.push_back(node);
    }
    
    std::vector<::onnx::TensorProto> reserve_tensors;
    reserve_tensors.reserve(graph->initializer_size());
    for (auto& tensor : graph->initializer()) {
        if (old_tensors.count(tensor.name())) {
            continue;
        }
        reserve_tensors.push_back(tensor);
    }
    
    graph->clear_node();
    graph->clear_initializer();
    
    for (auto& it : reserve_nodes) {
        *graph->add_node() = it;
    }
    for (auto& it : new_nodes) {
        *graph->add_node() = it;
    }
    for (auto& it : reserve_tensors) {
        *graph->add_initializer() = it;
    }
    for (auto& it : new_tensors) {
        *graph->add_initializer() = it;
    }
    return status;
}

}}} // namespace mariana::onnx::transform
