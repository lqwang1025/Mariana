/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/optimize/transform_utils.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:15:27:02
 * Description:
 *
 */

#ifndef __MARC_ONNX_OPTIMIZE_TRANSFORM_UTILS_H__
#define __MARC_ONNX_OPTIMIZE_TRANSFORM_UTILS_H__

#include <set>
#include <string>
#include <vector>

#include <core/utils/status.h>
#include <marc/onnx/onnx.h>

namespace mariana { namespace onnx { namespace transform {

struct OpTypePattern {
    std::string op;
    std::vector<OpTypePattern> inputs;
};

struct NodeMatch {
    NodeMatch() : node() {}
    ::onnx::NodeProto node;
    std::vector<NodeMatch> inputs;
};

class GraphMatcher {
public:
    GraphMatcher(OnnxScope& scope);

    // Sorts the input nodes into execution order, and then skips any previously
    // matches so that no node appears in more than one match. The NodeDef
    // pointers contained in the results are owned by the GraphMatcher object, and
    // so will be invalid after its lifetime.
    Status get_optype_matches(const OpTypePattern& pattern,
                              std::vector<NodeMatch>* matches);

private:
    bool _does_optype_match(const ::onnx::NodeProto& node, const OpTypePattern& pattern,
                            const std::set<std::string>& previously_matched_nodes,
                            NodeMatch* match);
    ::onnx::GraphProto graph_;
    OnnxScope& scope_;
    
};

}}} // namespace mariana::onnx::transform

#endif /* __MARC_ONNX_OPTIMIZE_TRANSFORM_UTILS_H__ */

