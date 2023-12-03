/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_concat.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-14:08:14:06
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/concat.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_concat_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    // MCHECK(inputs.size()==1)<<node->op_type()<<" support 1 input only.";
    std::vector<nvinfer1::ITensor *> input_trt_tensor_list;
    for (auto& it : inputs) {
        nvinfer1::ITensor* itensor = _get_itensor(it);
        nvinfer1::Dims dims = itensor->getDimensions();
        input_trt_tensor_list.push_back(itensor);
    }
    nvinfer1::IConcatenationLayer* layer = network_->addConcatenation(input_trt_tensor_list.data(), (int32_t)inputs.size());
    
    ConcatFunction* func = static_cast<ConcatFunction*>(node->op());
    layer->setAxis(func->option.axis);
    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
