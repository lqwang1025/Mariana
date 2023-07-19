/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_reshape.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:15:54:24
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/reshape.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_reshape_node(const Node& node, const ExecContext& context) {
    NodeList inputs = node.inputs();
    MCHECK(inputs.size()==1)<<node.op_type()<<" support 1 input only.";
    ReshapeFunction* func = static_cast<ReshapeFunction*>(node.op());
    
    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]->name());
    nvinfer1::IShuffleLayer* layer = network_->addShuffle(*itensor);
    nvinfer1::Dims dims;
    dims.nbDims = func->option.shape.size();
    for (size_t i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = func->option.shape[i];
    }
    std::cout<<"ss:"<<node.shapes()[0]<<std::endl;
    layer->setReshapeDimensions(dims);
    layer->setName(node.name().c_str());
    nvtensor_map_[node.name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
