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

bool TensorRTEngine::_add_eltwise_node(std::shared_ptr<Node>& node, const ConvertContext& context) {
    std::vector<std::string> inputs = node->inputs();

    auto eltwise_type_chose = [&]()->nvinfer1::ElementWiseOperation {
        MathFunction* func = static_cast<MathFunction*>(node->op());
        if (func->option.math_type == MathType::kSUM) {
            return nvinfer1::ElementWiseOperation::kSUM;
        } else if (func->option.math_type == MathType::kMUL) {
            return nvinfer1::ElementWiseOperation::kPROD;
        } else if (func->option.math_type == MathType::kDIV) {
            return nvinfer1::ElementWiseOperation::kDIV;
        } else if (func->option.math_type == MathType::kSUB) {
            return nvinfer1::ElementWiseOperation::kSUB;
        } else {
            MLOG(FATAL)<<"Unsupport act type:"<<node->op_type();
        }
    };
    
    if (inputs.size() == 2) {
        nvinfer1::ITensor* input1 = _get_itensor(inputs[0]);
        nvinfer1::ITensor* input2 = _get_itensor(inputs[1]);
        nvinfer1::IElementWiseLayer* layer = network_->addElementWise(*input1, *input2, eltwise_type_chose());
        layer->setName(node->name().c_str());
        nvtensor_map_[node->name()] = layer->getOutput(0);
    } else if (inputs.size() == 1) {
        nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
        MathFunction* func = static_cast<MathFunction*>(node->op());
        auto wshape = func->option.weight.shape();

        std::shared_ptr<Node> inode = inodes_of(node)[0];
        auto ishape = inode->shapes()[0];
        
        nvinfer1::Dims dims;
        dims.nbDims = ishape.dims();
        for (int32_t i = 0; i < dims.nbDims; ++i) {
            dims.d[i] = 1;
        }
        int32_t len_diff = ishape.dims()-wshape.dims();
        for (int32_t i = len_diff; i < ishape.dims(); ++i) {
            dims.d[i] = wshape[i-len_diff];
        }
        
        if (func->option.weight.dtype().match<float>()) {
            weights.values = func->option.weight.data();
            weights.count = func->option.weight.numel();
            weights.type = nvinfer1::DataType::kFLOAT; // Change me.
        } else {
            MLOG(FATAL)<<"Convolution support weight data type float now!";
        }
        nvinfer1::IConstantLayer* const_layer = network_->addConstant(dims, weights);
        std::string const_name = node->name()+"_constant";
        const_layer->setName(const_name.c_str());
        nvtensor_map_[const_name.c_str()] = const_layer->getOutput(0);
        
        nvinfer1::ITensor* input = _get_itensor(inputs[0]);
        nvinfer1::IElementWiseLayer* layer = network_->addElementWise(*input, *const_layer->getOutput(0), eltwise_type_chose());
        layer->setName(node->name().c_str());
        nvtensor_map_[node->name()] = layer->getOutput(0);
        
        auto dim = layer->getOutput(0)->getDimensions();
        
    } else {
        MLOG(FATAL)<<"Unsupport eltwise op input size:"<<inputs.size()<<" in "<<node->name();
    }
    
    return true;
}

}} // namespace mariana::trt
