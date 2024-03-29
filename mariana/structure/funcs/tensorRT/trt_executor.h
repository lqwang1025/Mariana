/*
 *        (C) COPYRIGHT Daniel Limited.
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
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>

#include <structure/ir.h>
#include <structure/engine.h>
#include <core/utils/status.h>
#include <structure/funcs/ops.h>
#include <structure/graph_exec.h>

namespace mariana { namespace trt {

// There are two ways to build a TRT engine:
//   one is to build NetWork directly, and the other is to build it through TRT's APIs.
class TensorRTEngine : public ::mariana::Engine {
public:
    using TrtLayerMake = std::function<bool(TensorRTEngine*, std::shared_ptr<Node>&, const proto::ModelInfo&)>;
    TensorRTEngine();
    ~TensorRTEngine();
    Status build_external(Graph& graph, const proto::ModelInfo& model_info) override;
    Status build_internal(Graph& graph, const proto::ModelInfo& model_info) override;
    Status de_serialize(Graph& graph, const proto::ModelInfo& model_info) override;
    Status run(const ExecContext& context) override;
private:
    void _setup_optimize(const proto::ModelInfo& model_info);
    nvinfer1::ITensor* _get_itensor(const std::string& iname);
    nvinfer1::ITensor* _add_input(const Shape& shape, const std::string& name, nvinfer1::DataType type);
    Status _construct_network(Graph& graph, const proto::ModelInfo& model_info);
private:
    bool _add_convolution_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_act_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_pool_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_eltwise_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_reshape_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_softmax_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_fc_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_reduce_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_slice_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_concat_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_resize_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);
    bool _add_transpose_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info);    
private:
    const std::string input_prefix_ = "_minput";
    std::unique_ptr<nvinfer1::IBuilder> builder_ = nullptr;
    std::unique_ptr<nvinfer1::INetworkDefinition> network_ = nullptr;
    std::unique_ptr<nvinfer1::IBuilderConfig> config_ = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    cudaStream_t stream_;
    std::unordered_map<std::string, TrtLayerMake> layer_make_map_ = {
        {MCONV2D, &TensorRTEngine::_add_convolution_node},
        {MRELU, &TensorRTEngine::_add_act_node},
        {MSIGMOID, &TensorRTEngine::_add_act_node},
        {MMAXPOOL, &TensorRTEngine::_add_pool_node},
        {MADD, &TensorRTEngine::_add_eltwise_node},
        {MSUB, &TensorRTEngine::_add_eltwise_node},
        {MDIV, &TensorRTEngine::_add_eltwise_node},
        {MMUL, &TensorRTEngine::_add_eltwise_node},
        {MGAVPOOL, &TensorRTEngine::_add_pool_node},
        {MRESHAPE, &TensorRTEngine::_add_reshape_node},
        {MSOFTMAX, &TensorRTEngine::_add_softmax_node},
        {MGEMM, &TensorRTEngine::_add_fc_node},
        {MREDUCEMEAN, &TensorRTEngine::_add_reduce_node},
        {MSLICE, &TensorRTEngine::_add_slice_node},
        {MCONCAT, &TensorRTEngine::_add_concat_node},
        {MRESIZE, &TensorRTEngine::_add_resize_node},
        {MPERMUTE, &TensorRTEngine::_add_transpose_node},
        {MFLATTEN, &TensorRTEngine::_add_reshape_node},
    };
    std::unordered_map<std::string, nvinfer1::ITensor*> nvtensor_map_;
};

}} // namespace mariana::trt

#endif /* __STRUCTURE_FUNCS_TENSORRT_TRT_EXECUTOR_H__ */

