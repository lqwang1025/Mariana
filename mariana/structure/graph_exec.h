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
#include <core/device.h>
#include <structure/tensor.h>

namespace mariana {

struct ExecContext {
    std::unordered_map<std::string, Shape> ishapes;
    std::unordered_map<std::string, Tensor> itensors;
    Device device;
};

class GraphExec final {
public:
    GraphExec() {}
    ~GraphExec() {}
    void run(Graph& graph, ExecContext& context);
    void pre_run(Graph& graph, ExecContext& context);
};

} // namespace mariana

#endif /* __STRUCTURE_GRAPH_EXEC_H__ */

 
