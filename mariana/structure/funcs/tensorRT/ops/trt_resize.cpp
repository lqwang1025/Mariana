/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_resize.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-14:08:29:27
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/resize.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_resize_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    MCHECK(inputs.size()==1)<<node->op_type()<<" support 1 input only.";

    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]);
    nvinfer1::IResizeLayer* layer = network_->addResize(*itensor);
    MCHECK(layer!=nullptr)<<"Mariana: create layer slice failed!";
    // nvinfer1::Dims odim;
    // odim.nbDims = node->shapes()[0].dims();
    // for (uint8_t i = 0; i < node->shapes()[0].dims(); ++i) {
    //     odim.d[i] = node->shapes()[0][i];
    // }
    // layer->setOutputDimensions(odim);
    
    ResizeFunction* func = static_cast<ResizeFunction*>(node->op());
    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kLINEAR;
    if (func->option.resize_mode == ResizeMode::Nearest) {
        resizeMode = nvinfer1::ResizeMode::kNEAREST;
    }
    layer->setResizeMode(resizeMode);
    layer->setScales(func->option.scales.data(), func->option.scales.size());
    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
