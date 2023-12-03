/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_softmax.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:15:55:29
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/softmax.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_softmax_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    MCHECK(inputs.size()==1)<<node->op_type()<<" support 1 input only.";
    SoftmaxFunction* func = static_cast<SoftmaxFunction*>(node->op());
    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]);

    nvinfer1::ISoftMaxLayer* layer = network_->addSoftMax(*itensor);
    uint32_t axes = 1u << (func->option.axis-static_cast<int>(network_->hasImplicitBatchDimension()));
    layer->setAxes(axes);
    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
