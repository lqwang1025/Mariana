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

#include <map>
#include <functional>
#include <unordered_map>
#include <string>

#include <structure/ir.h>
#include <core/utils/status.h>
#include <structure/graph_exec.h>

namespace mariana { namespace trt {

class TensorRTEngine {
public:
    using TrtLayerMake = std::function<bool(TensorRTEngine*, const Node&, const ExecContext&)>;
    TensorRTEngine();
    ~TensorRTEngine() = default;
    Status pre_run(const Graph& graph, const ExecContext& context);
private:
    nvinfer1::ITensor* _add_tensor(const Shape& shape, const std::string& name, nvinfer1::DataType type);
    Status _build(const Graph& graph, const ExecContext& context);
    bool _add_convolution_node(const Node& node, const ExecContext& context);
    // bool _add_act_node(const Node& node, const ExecContext& context);
    // bool _add_pool_node(const Node& node, const ExecContext& context);
    // bool _add_add_node(const Node& node, const ExecContext& context);
    
private:
    nvinfer1::IBuilder* builder_;
    nvinfer1::INetworkDefinition* network_;
    nvinfer1::IBuilderConfig* config_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::unordered_map<std::string, TrtLayerMake> layer_make_map_ = {
        {"Conv", &TensorRTEngine::_add_convolution_node},
        // {"Relu", &TensorRTEngine::_add_act_node},
        // {"SoftMax", &TensorRTEngine::_add_act_node},
        // {"MaxPool", &TensorRTEngine::_add_pool_node},
        // {"GlobalAveragePool", &TensorRTEngine::_add_pool_node},
        // {"Add", &TensorRTEngine::_add_add_node}
    };
};

}} // namespace mariana::trt

#endif /* __STRUCTURE_FUNCS_TENSORRT_TRT_EXECUTOR_H__ */

