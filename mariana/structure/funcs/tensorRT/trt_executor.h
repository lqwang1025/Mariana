/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/tensorRT/trt_executor.h
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:11:13:06
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_TENSORRT_TRT_EXECUTOR_H__
#define __STRUCTURE_FUNCS_TENSORRT_TRT_EXECUTOR_H__

#include <NvInfer.h>

#include <structure/ir.h>

namespace mariana { namespace trt {

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine() = default;
private:
    int _build(const Graph& graph);
private:
    nvinfer1::IBuilder* builder;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
};

}} // namespace mariana::trt

#endif /* __STRUCTURE_FUNCS_TENSORRT_TRT_EXECUTOR_H__ */

