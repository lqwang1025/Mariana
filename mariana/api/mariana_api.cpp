/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : mariana_api.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:09:48:55
 * Description:
 * 
 */

#include <mariana_api.h>
#include <marc/marc.h>
#include <core/utils/logging.h>
#include <structure/ir.h>
#include <structure/graph_exec.h>

namespace mariana {

Runtime::Runtime(const ConvertContext& ccontext) {
    mariana::Graph* graph = mariana::parse(ccontext);
    if (graph->engine()) {
        for (auto &it : graph->engine()->itensors) {
            input_names.push_back(it.name());
        }
        for (auto &it : graph->engine()->otensors) {
            output_names.push_back(it.name());
        }
    } else {
        MCHECK(false);
    }
    handle_ = graph;
}

Runtime::~Runtime() {
    mariana::Graph* graph = static_cast<mariana::Graph*>(handle_);
    delete graph;
}

std::vector<MResult> Runtime::run_with(ExecContext& econtext) {
    mariana::GraphExec ge;
    mariana::Graph* graph = static_cast<mariana::Graph*>(handle_);
    ge.run(*graph, econtext);
    return ge.results;
}

} // namespace mariana
