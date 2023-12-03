/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_fc.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:16:43:22
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/gemm.h>
#include <structure/tensor.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_fc_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    MCHECK(inputs.size()<2)<<node->op_type()<<" support 1 input only.";
    
    std::string itname;
    if (inputs.size() == 0) {
        itname = node->name()+input_prefix_;
    } else if (inputs.size() == 1) {
        itname = inputs[0];
    }
    nvinfer1::ITensor* itensor = _get_itensor(itname);
    GemmFunction* func = static_cast<GemmFunction*>(node->op());
    
    const std::vector<Tensor>& weights = func->option.weights;
    nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, 0};
    if (weights[0].dtype().match<float>()) {
        weight.values = weights[0].data();
        weight.count = weights[0].numel();
        weight.type = nvinfer1::DataType::kFLOAT;
    } else {
        MLOG(FATAL)<<"Convolution support weight data type float now!";
    }
    
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    if (weights.size() == 2) {
        if (weights[1].dtype().match<float>()) {
            bias.values = weights[1].data();
            bias.count = weights[1].numel();
            bias.type = nvinfer1::DataType::kFLOAT;
        } else {
            MCHECK(false);
        }
    }
    int oc = weights[0].shape()[0];
    nvinfer1::IFullyConnectedLayer* layer = network_->addFullyConnected(*itensor, oc, weight, bias);

    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
