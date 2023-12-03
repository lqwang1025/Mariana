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

bool TensorRTEngine::_add_slice_node(std::shared_ptr<Node>& node, const proto::ModelInfo& model_info) {
    std::vector<std::string> inputs = node->inputs();
    MCHECK(inputs.size()==1)<<node->op_type()<<" support 1 input only.";

    std::shared_ptr<Node> inode = inodes_of(node)[0];
    
    nvinfer1::Dims start, size, stride;
    start.nbDims  = inode->shapes()[0].dims();
    size.nbDims   = inode->shapes()[0].dims();
    stride.nbDims = inode->shapes()[0].dims();
    
    for (uint8_t i = 0; i < inode->shapes()[0].dims(); ++i) {
        start.d[i] = 0;
        size.d[i] = node->shapes()[0][i];
        stride.d[i] = 1;
    }
    SliceFunction* func = static_cast<SliceFunction*>(node->op());
    start.d[func->option.axis] = func->option.begin;
    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]);
    nvinfer1::ISliceLayer* layer = network_->addSlice(*itensor, start, size, stride);
    MCHECK(layer!=nullptr)<<"Mariana: create layer slice failed!";
    layer->setName(node->name().c_str());
    nvtensor_map_[node->name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
