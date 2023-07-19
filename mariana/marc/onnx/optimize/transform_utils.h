/*
 *        (C) COPYRIGHT Daniel Limited.
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
#include <map>
#include <string>
#include <vector>

#include <core/utils/status.h>
#include <marc/onnx/onnx.h>

namespace mariana { namespace onnx { namespace transform {

struct OpTypePattern {
    std::string op;
    std::vector<OpTypePattern> inputs;
    std::string debug_string() const;
};

struct NodeMatch {
    NodeMatch() : node() {}
    ::onnx::NodeProto node;
    std::vector<NodeMatch> inputs;
    std::string debug_string() const;
};

Status replace_matching_optypes(OnnxScope& scope, const OpTypePattern& pattern,
                                const std::function<Status(OnnxScope& scope, const NodeMatch&, std::set<std::string>*, std::set<std::string>*, std::vector<::onnx::NodeProto>*, std::vector<::onnx::TensorProto>*)>& node_generator,
                                ::onnx::GraphProto* graph);

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

typedef std::function<Status(OnnxScope& scope)> TransformFunc;
typedef std::map<std::string, TransformFunc> TransformRegistry;
TransformRegistry* get_transform_registry();

class TransformRegistrar {
public:
    TransformRegistrar(const std::string& name, TransformFunc transform_func) {
        TransformRegistry* transform_registry = get_transform_registry();
        (*transform_registry)[name] = transform_func;
    }
};

#define REGISTER_ONNX_GRAPH_TRANSFORM(name, func)                       \
    REGISTER_ONNX_GRAPH_TRANSFORM_UNIQ_HELPER(__COUNTER__, name, func)
#define REGISTER_ONNX_GRAPH_TRANSFORM_UNIQ_HELPER(ctr, name, func)  \
    REGISTER_ONNX_GRAPH_TRANSFORM_UNIQ(ctr, name, func)
#define REGISTER_ONNX_GRAPH_TRANSFORM_UNIQ(ctr, name, func) \
    static mariana::onnx::transform::TransformRegistrar     \
    registrar__body__##ctr##__object(name, func);

}}} // namespace mariana::onnx::transform

#endif /* __MARC_ONNX_OPTIMIZE_TRANSFORM_UTILS_H__ */

