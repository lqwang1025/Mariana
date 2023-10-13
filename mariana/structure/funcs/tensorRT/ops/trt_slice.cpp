/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_slice.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-08:09:04:57
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/slice.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_slice_node(const Node& node, const ConvertContext& context) {
    NodeList inputs = node.inputs();
    MCHECK(inputs.size()==1)<<node.op_type()<<" support 1 input only.";
    SliceFunction* func = static_cast<SliceFunction*>(node.op());
    
    nvinfer1::Dims start, size, stride;
    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]->name());
    // int32_t p = 0;
    // for (auto& it : node.output_edges()) {

    //     start.nbDims  = inputs[0]->shapes()[0].dims();
    //     size.nbDims   = inputs[0]->shapes()[0].dims();
    //     stride.nbDims = inputs[0]->shapes()[0].dims();
    //     int32_t ctrl_index = it.get_ctrl_index();
    //     for (uint8_t i = 0; i < inputs[0]->shapes()[0].dims(); ++i) {
    //         start.d[i] = 0;
    //         size.d[i] = node.shapes()[ctrl_index][i];
    //         stride.d[i] = 1;
    //     }
    //     start.d[func->option.axis] = splits[ctrl_index];
    //     nvinfer1::ISliceLayer* layer = network_->addSlice(*itensor, start, size, stride);
    //     MCHECK(layer!=nullptr)<<"Mariana: create layer slice failed!";
    //     std::string layer_name = node.name() + std::to_string(p++);
    //     layer->setName(layer_name.c_str());
    //     nvtensor_map_[layer_name] = layer->getOutput(0);
    //     std::cout<<"layer_name:"<<layer_name<<std::endl;
    // }
    return true;
}

}} // namespace mariana::trt
