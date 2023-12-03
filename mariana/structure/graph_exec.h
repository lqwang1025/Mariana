/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/graph_exec.h
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:16:18:50
 * Description:
 *
 */

#ifndef __STRUCTURE_GRAPH_EXEC_H__
#define __STRUCTURE_GRAPH_EXEC_H__

#include <string>
#include <unordered_map>

#include <structure/ir.h>
#include <structure/tensor.h>
#include <api/proto/mariana.pb.h>

namespace mariana {

class GraphExec final {
public:
    GraphExec() {}
    ~GraphExec() {}
    void run(Graph& graph, ExecContext& context);
    void pre_run(Graph& graph, ExecContext& context);
    void pre_run(Graph& graph, const proto::ModelInfo& model_info);
    MResult results;
};

} // namespace mariana

#endif /* __STRUCTURE_GRAPH_EXEC_H__ */

 
