/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/tensorRT/trt_executor.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:11:16:15
 * Description:
 * 
 */

#include <NvInferRuntime.h>
#include <logging.h>
#include <structure/funcs/tensorRT/trt_executor.h>

namespace mariana { namespace trt {

static ::sample::Logger gLogger{};

TensorRTEngine::TensorRTEngine() {}

Status TensorRTEngine::_build(const Graph& graph) {
    builder_ = nvinfer1::createInferBuilder(gLogger);
    network_ = builder_->createNetwork();
    for (auto& node : graph.nodes()) {
        std::cout<<"debi:"<<node->op_type()<<std::endl;
    }
    return absl::OkStatus();
}

}} // namespace mariana::trt
