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

bool TensorRTEngine::_add_convolution_node(const Node& node, const ExecContext& context) {
    std::cout<<"gfffffffffffffffffffffff"<<std::endl;
    NodeList inputs = node.inputs();
    // std::cout<<"deug:"<<inputs[0]<<std::endl;
    std::cout<<"debug:"<<inputs.size()<<std::endl;
    nvinfer1::ITensor* itensor;
    if (inputs.size() == 0) {
        const Shape& ishape = context.ishapes.at(node.name());
        itensor = _add_tensor(ishape, node.name()+"_input", nvinfer1::DataType::kFLOAT);
    } else if (inputs.size() == 1) {
        std::cout<<"de:"<<inputs[0]->name()<<std::endl;
        const Shape& ishape = inputs[0]->shapes()[0];
        std::cout<<"de:"<<inputs[0]->shapes().size()<<std::endl;
        itensor = _add_tensor(ishape, inputs[0]->name(), nvinfer1::DataType::kFLOAT);
    } else {
        MCHECK(false);
    }
    ConvFunction* func = static_cast<ConvFunction*>(node.op());
    
    const std::vector<Tensor>& weights = func->option.weights;
    nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, 0};
    if (weights[0].dtype() == TypeMeta::make<float>()) {
        weight.values = weights[0].data();
        weight.count = weights[0].numel();
        weight.type = nvinfer1::DataType::kFLOAT;
    } else {
        MCHECK(false);
    }
    
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    if (weights.size() == 2) {
        if (weights[1].dtype() == TypeMeta::make<float>()) {
            bias.values = weights[1].data();
            bias.count = weights[1].numel();
            bias.type = nvinfer1::DataType::kFLOAT;
        } else {
            MCHECK(false);
        }
    }
    nvinfer1::DimsHW kernel_size{func->option.kernel_shape[0], func->option.kernel_shape[1]};
    int oc = func->option.oc;
    nvinfer1::IConvolutionLayer* layer = network_->addConvolution(*itensor, oc, kernel_size, weight, bias);

    layer->setStride(nvinfer1::DimsHW(func->option.strides[0], func->option.strides[1]));
    layer->setPrePadding(nvinfer1::DimsHW(func->option.pads[0], func->option.pads[2]));
    layer->setPostPadding(nvinfer1::DimsHW(func->option.pads[1], func->option.pads[3]));
    layer->setDilation(nvinfer1::DimsHW(func->option.dilations[0], func->option.dilations[1]));
    layer->setNbGroups(func->option.group);

    layer->setName(node.name().c_str());
    
    return true;
}

}} // namespace mariana::trt
