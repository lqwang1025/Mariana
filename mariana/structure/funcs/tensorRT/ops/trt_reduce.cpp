/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_reduce.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-25:10:17:21
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>
#include <structure/funcs/reduce.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_reduce_node(const Node& node, const ConvertContext& context) {
    NodeList inputs = node.inputs();
    MCHECK(inputs.size()==1)<<node.op_type()<<" support 1 input only.";
    ReduceFunction* func = static_cast<ReduceFunction*>(node.op());

    auto reduce_type_chose = [&]()->nvinfer1::ReduceOperation {
        if (func->option.method == ReduceMethod::SUM) {
            return nvinfer1::ReduceOperation::kSUM;
        } else if (func->option.method == ReduceMethod::MEAN) {
            return nvinfer1::ReduceOperation::kAVG;
        } else {
            MLOG(FATAL)<<"Unsupport reduce type:"<<node.op_type();
        }
    };
    uint32_t reduceAxes = 0;
    for (auto& it : func->option.axes) {
        int flag = 1;
        flag = flag << (it-1);
        reduceAxes += flag;
    }
    nvinfer1::ITensor* itensor = _get_itensor(inputs[0]->name());
    nvinfer1::IReduceLayer* layer = network_->addReduce(*itensor, reduce_type_chose(), reduceAxes, func->option.keepdims);
    layer->setName(node.name().c_str());
    nvtensor_map_[node.name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt

