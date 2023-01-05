/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : maro/transform_utils.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:11:37:37
 * Description:
 * 
 */

#include <absl/strings/str_split.h>
#include <maro/transform_utils.h>
#include <core/utils/logging.h>

namespace mariana { namespace transform {

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

GraphMatcher::GraphMatcher(Graph& graph) : graph_(graph), scope_(&graph) {}

bool GraphMatcher::_does_optype_match(const Node& node, const OpTypePattern& pattern,
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
    for (auto& input : node.relationships().input_edges) {
        if (!input.get_node().name().empty()) {
            non_control_inputs.push_back(input.get_node().name());
        }
    }
    if (non_control_inputs.size() != pattern.inputs.size()) {
        MVLOG(1) << "non_control_inputs.size() != pattern.inputs.size() "
                 << non_control_inputs.size()<<" : "<<pattern.inputs.size();
        return false;
    }
    for (size_t i = 0; i < pattern.inputs.size(); ++i) {
        const std::string& input_node_name = non_control_inputs[i];
        const Node& input_node = *(scope_.node_name_map[input_node_name]);
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
    for (auto& node : graph_.nodes()) {
        // Skip any nodes that are already part of a match.
        if (matched_nodes.count(node->name())) {
            continue;
        }
        NodeMatch match;
        if (_does_optype_match(*node, pattern, matched_nodes, &match)) {
            record_matched_nodes(match, &matched_nodes);
            matches->push_back(match);
        }
    }
    return absl::OkStatus();
}

Status replace_matching_optypes(Graph& src, const OpTypePattern& pattern,
                                const std::function<Status(Scope& scope, const NodeMatch&, std::set<std::string>*, std::vector<Node>*)>& node_generator, Graph* dst) {
    GraphMatcher matcher{src};
    std::vector<NodeMatch> matches;
    Status res = matcher.get_optype_matches(pattern, &matches);
    if (!res.ok()) {
        MLOG(ERROR) << "get_optype_matches error";
        return res;
    }
    Status status = absl::OkStatus();
    std::set<std::string> old_nodes;
    std::vector<Node> new_nodes;
    for (const NodeMatch& match : matches) {
        std::vector<Node> _new_nodes;
        status = node_generator(matcher.scope(), match, &old_nodes, &_new_nodes);
        new_nodes.insert(new_nodes.end(), _new_nodes.begin(), _new_nodes.end());
    }
    std::vector<Node> reserve_nodes;
    reserve_nodes.reserve(src.num_of_nodes());
    for (auto& node : src.nodes()) {
        if (old_nodes.count(node->name())) {
            continue;
        }
        reserve_nodes.push_back(*node);
    }
    for (auto& it : reserve_nodes) {
        *dst->make_node() = it;
    }
    for (auto& it : new_nodes) {
        *dst->make_node() = it;
    }
    Scope::sort_by_exe_order(dst);
    return status;
}

TransformRegistry* get_transform_registry() {
    static TransformRegistry transform_registry;
    return &transform_registry;
}

}} // namespace mariana::transform
