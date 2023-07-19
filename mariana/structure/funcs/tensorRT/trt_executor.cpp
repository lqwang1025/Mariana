/*
 *        (C) COPYRIGHT Daniel Limited.
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
#include <structure/funcs/tensorRT/trt_helper.h>

namespace mariana { namespace trt {

static ::sample::Logger gLogger{};

TensorRTEngine::TensorRTEngine() {}

Status TensorRTEngine::pre_run(const Graph& graph, const ExecContext& context) {
    return _build(graph, context);
}

nvinfer1::ITensor* TensorRTEngine::_get_itensor(const std::string& iname) {
    if (nvtensor_map_.count(iname)) {
        return nvtensor_map_[iname];
    } else {
        MLOG(FATAL)<<"Can not find the input tensor:"<<iname<<" construct trt engine failed!!";
    }
}

Status TensorRTEngine::_build(const Graph& graph, const ExecContext& context) {
    builder_.reset(nvinfer1::createInferBuilder(gLogger));
    const auto _explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network_.reset(builder_->createNetworkV2(_explicit_batch));
    for (auto& it : context.ishapes) { // set network inputs
        std::string itname = it.first+input_prefix_;
        nvtensor_map_[itname] = _add_input(it.second, itname, nvinfer1::DataType::kFLOAT);
    }
    
    for (auto& node : graph.nodes()) {
        if (layer_make_map_.count(node->op_type())) {
            layer_make_map_[node->op_type()](this, *node, context);
        }
        if (node->outputs().size() == 0) { // set network outputs
            network_->markOutput(*_get_itensor(node->name()));
        }
    }
    config_.reset(builder_->createBuilderConfig());
    return absl::OkStatus();
}

nvinfer1::ITensor* TensorRTEngine::_add_input(const Shape& shape, const std::string& name, nvinfer1::DataType type) {
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
