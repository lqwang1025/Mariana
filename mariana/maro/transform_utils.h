/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : maro/transform_utils.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:15:11:36
 * Description:
 *
 */

#ifndef __MARO_TRANSFORM_UTILS_H__
#define __MARO_TRANSFORM_UTILS_H__

#include <map>
#include <string>
#include <vector>

#include <structure/ir.h>
#include <core/utils/status.h>

namespace mariana { namespace transform {

struct OpTypePattern {
    std::string op;
    std::vector<OpTypePattern> inputs;
    std::string debug_string() const;
};

struct NodeMatch {
    NodeMatch() : node() {}
    std::shared_ptr<Node> node = nullptr;
    std::vector<NodeMatch> inputs;
    std::string debug_string() const;
};

Status replace_matching_optypes(Graph& src, const OpTypePattern& pattern,
                                const std::function<Status(const NodeMatch&, std::set<std::string>*, std::vector<std::shared_ptr<Node>>*)>& node_generator);

class GraphMatcher {
public:
    GraphMatcher(Graph& graph);

    // Sorts the input nodes into execution order, and then skips any previously
    // matches so that no node appears in more than one match. The NodeDef
    // pointers contained in the results are owned by the GraphMatcher object, and
    // so will be invalid after its lifetime.
    Status get_optype_matches(const OpTypePattern& pattern,
                              std::vector<NodeMatch>* matches);
private:
    bool _does_optype_match(std::shared_ptr<Node>& node, const OpTypePattern& pattern,
                            const std::set<std::string>& previously_matched_nodes,
                            NodeMatch* match);
    Graph graph_;
    
};

typedef std::function<Status(Graph& graph)> TransformFunc;
typedef std::map<std::string, TransformFunc> TransformRegistry;
TransformRegistry* get_transform_registry();

class TransformRegistrar {
public:
    TransformRegistrar(const std::string& name, TransformFunc transform_func) {
        TransformRegistry* transform_registry = get_transform_registry();
        (*transform_registry)[name] = transform_func;
    }
};

#define REGISTER_GRAPH_TRANSFORM(name, func)                        \
    REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(__COUNTER__, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(ctr, name, func)   \
    REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)  \
    static mariana::transform::TransformRegistrar       \
    registrar__body__##ctr##__object(name, func);


}} // namespace mariana::transform

#endif /* __MARO_TRANSFORM_UTILS_H__ */

