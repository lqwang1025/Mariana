/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_eltwise.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:13:48:51
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_executor.h>
#include <core/utils/logging.h>
#include <structure/funcs/math.h>
#include <structure/funcs/ops.h>
#include <structure/tensor.h>

namespace mariana { namespace trt {

bool TensorRTEngine::_add_eltwise_node(const Node& node, const ConvertContext& context) {
    NodeList inputs = node.inputs();
    MCHECK(inputs.size()==2)<<node.op_type()<<" support 2 inputs only.";
    
    auto eltwise_type_chose = [&]()->nvinfer1::ElementWiseOperation {
        MathFunction* func = static_cast<MathFunction*>(node.op());
        if (func->option.math_type == MathType::kSUM) {
            return nvinfer1::ElementWiseOperation::kSUM;
        } else if (func->option.math_type == MathType::kMUL) {
            return nvinfer1::ElementWiseOperation::kPROD;
        } else {
            MLOG(FATAL)<<"Unsupport act type:"<<node.op_type();
        }
    };
    nvinfer1::ITensor* input1 = _get_itensor(inputs[0]->name());
    nvinfer1::ITensor* input2 = _get_itensor(inputs[1]->name());
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(*input1, *input2, eltwise_type_chose());
    layer->setName(node.name().c_str());
    nvtensor_map_[node.name()] = layer->getOutput(0);
    return true;
}

}} // namespace mariana::trt
