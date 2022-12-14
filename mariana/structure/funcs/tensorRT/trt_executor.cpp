/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/tensorRT/trt_executor.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:11:16:15
 * Description:
 * 
 */

#include <NvInferRuntime.h>
#include <logging.h>

#include <structure/ir.h>
#include <core/utils/logging.h>
#include <structure/funcs/tensorRT/trt_executor.h>

namespace mariana { namespace trt {

static ::sample::Logger gLogger{};

TensorRTEngine::TensorRTEngine() {}

Status TensorRTEngine::pre_run(const Graph& graph, const ExecContext& context) {
    return _build(graph, context);
}

Status TensorRTEngine::_build(const Graph& graph, const ExecContext& context) {
    builder_ = nvinfer1::createInferBuilder(gLogger);
    network_ = builder_->createNetwork();
    for (auto& node : graph.nodes()) {
        std::cout<<"debug:"<<node->name()<<std::endl;
        for (auto& input : node->inputs()) {
            std::cout<<"------>input:"<<input->name()<<std::endl;
        }
    }
    // for (auto& node : graph.nodes()) {
    //     std::cout<<"debug:"<<node->name()<<std::endl;
    //     if (layer_make_map_.count(node->op_type()))
    //         layer_make_map_[node->op_type()](this, *node, context);
    // }
    return absl::OkStatus();
}

nvinfer1::ITensor* TensorRTEngine::_add_tensor(const Shape& shape, const std::string& name, nvinfer1::DataType type) {
    int32_t dim = shape.dims();
    nvinfer1::ITensor* trt_tensor;
    switch (dim) {
    case 2: {
        nvinfer1::Dims2 dim2(static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[1]));
        trt_tensor = this->network_->addInput(name.c_str(), type, dim2);
        break;
    }
    case 3: {
        nvinfer1::Dims3 dim3(static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[1]), static_cast<int32_t>(shape[2]));
        trt_tensor = this->network_->addInput(name.c_str(), type, dim3);
        break;
    }
    case 4: {
        nvinfer1::Dims4 dim4(static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[1]),
                             static_cast<int32_t>(shape[2]), static_cast<int32_t>(shape[3]));
        trt_tensor = this->network_->addInput(name.c_str(), type, dim4);
        break;
    }
    default: {
        MCHECK(false)<<"Tensor dimension"<<dim<<" cannot be supported.";
        return nullptr;
    }
    }
    return trt_tensor;
}

}} // namespace mariana::trt
