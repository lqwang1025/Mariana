/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_pool.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:11:44:27
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/pool.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_pool_node(const Node& node, const ConvertContext& context) {
    std::vector<std::string> inputs = node.inputs();
    MCHECK(inputs.size()==1)<<node.op_type()<<" support 1 input only.";
    PoolFunction* func = static_cast<PoolFunction*>(node.op());
    nvinfer1::Dims kernel_size = {.nbDims = 2,
                                  .d = {func->option.kernel_shape[0],
                                        func->option.kernel_shape[1]}};
    auto pool_type_chose = [&]()->nvinfer1::PoolingType {
        if (func->option.type == PoolType::Max) {
            return nvinfer1::PoolingType::kMAX;
        } else if (func->option.type == PoolType::Avg || func->option.type == PoolType::GAvg) {
            return nvinfer1::PoolingType::kAVERAGE;
        } else {
            MLOG(FATAL)<<"Unsupport pooling type:"<<node.op_type();
        }
    };

    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]);
    nvinfer1::IPoolingLayer* layer = network_->addPoolingNd(*itensor, pool_type_chose(), kernel_size);
    layer->setStride(nvinfer1::DimsHW{func->option.strides[0], func->option.strides[1]});
    layer->setAverageCountExcludesPadding(false);
    layer->setPrePadding(nvinfer1::DimsHW(func->option.pads[0], func->option.pads[1]));
    layer->setPostPadding(nvinfer1::DimsHW(func->option.pads[2], func->option.pads[3]));
    layer->setName(node.name().c_str());
    nvtensor_map_[node.name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
