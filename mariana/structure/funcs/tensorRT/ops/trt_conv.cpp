/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_conv.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-05:15:43:50
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/conv.h>
#include <structure/tensor.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_convolution_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    MCHECK(inputs.size()<2)<<node->op_type()<<" support 1 input only.";
    
    std::string itname;
    if (inputs.size() == 0) {
        itname = node->name()+input_prefix_;
    } else if (inputs.size() == 1) {
        itname = inputs[0];
    }
    nvinfer1::ITensor* itensor = _get_itensor(itname);
    ConvFunction* func = static_cast<ConvFunction*>(node->op());
    
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
    nvinfer1::DimsHW kernel_size{func->option.kernel_shape[0], func->option.kernel_shape[1]};
    int oc = func->option.oc;
    nvinfer1::IConvolutionLayer* layer = network_->addConvolutionNd(*itensor, oc, kernel_size, weight, bias);

    layer->setStride(nvinfer1::DimsHW(func->option.strides[0], func->option.strides[1]));
    layer->setPrePadding(nvinfer1::DimsHW(func->option.pads[0], func->option.pads[1]));
    layer->setPostPadding(nvinfer1::DimsHW(func->option.pads[2], func->option.pads[3]));
    layer->setDilation(nvinfer1::DimsHW(func->option.dilations[0], func->option.dilations[1]));
    layer->setNbGroups(func->option.group);

    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);


    auto dims = layer->getOutput(0)->getDimensions();
    auto dim = itensor->getDimensions();
    
    return true;
}

}} // namespace mariana::trt
