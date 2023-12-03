/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_transpose.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-16:08:58:47
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/permute.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_transpose_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    MCHECK(inputs.size()==1)<<node->op_type()<<" support 1 input only.";
    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]);
    
    nvinfer1::IShuffleLayer* layer = network_->addShuffle(*itensor);
    
    PermuteFunction* func = static_cast<PermuteFunction*>(node->op());
    std::vector<int32_t> perm = func->option.perm;
    nvinfer1::Permutation order = { 0 };
    for (size_t i = 0; i < perm.size(); i++) {
        order.order[i] = perm[i];
    }
    
    for (size_t i = perm.size(); i < nvinfer1::Dims::MAX_DIMS; i++) {
        order.order[i] = 0;
    }
    layer->setZeroIsPlaceholder(false);
    layer->setFirstTranspose(order);
    
    nvinfer1::Dims dims{};
    dims.nbDims = node->shapes()[0].dims();
    for (int i = 0; i < dims.nbDims; i++)
        dims.d[i] = node->shapes()[0][i];
    
    layer->setReshapeDimensions(dims);
    
    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
