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
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <logging.h>
#include <fstream>
#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <structure/ir.h>
#include <core/utils/logging.h>
#include <structure/funcs/tensorRT/trt_executor.h>
#include <structure/funcs/tensorRT/trt_helper.h>

namespace mariana { namespace trt {

static ::sample::Logger gLogger{};

TensorRTEngine::TensorRTEngine() {}

TensorRTEngine::~TensorRTEngine() {
    cudaStreamDestroy(stream_);
}

Status TensorRTEngine::build_external(Graph& graph, const ConvertContext& context) {
    builder_.reset(nvinfer1::createInferBuilder(gLogger));
    const auto _explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    _setup_optimize(context); // config init
    network_.reset(builder_->createNetworkV2(_explicit_batch));
    auto parser = nvonnxparser::createParser(*network_, gLogger);
    parser->parseFromFile(context.model_path.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    builder_->setMaxBatchSize(context.max_batch_size);
    nvinfer1::IHostMemory* hm = builder_->buildSerializedNetwork(*network_, *config_);
    std::ofstream serialize_output_stream;
    std::string output_path = absl::StrReplaceAll(context.model_path, {{".onnx", ".plan"}});
    serialize_output_stream.open(output_path, std::ios::out|std::ios::binary);
    serialize_output_stream.write((const char*)hm->data(), hm->size());
    serialize_output_stream.close();
    parser->destroy();
    ConvertContext __context;
    __context.model_path = output_path;
    de_serialize(graph, __context);
    return absl::OkStatus();
}

Status TensorRTEngine::build_internal(Graph& graph, const ConvertContext& context) {
    builder_.reset(nvinfer1::createInferBuilder(gLogger));
    const auto _explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network_.reset(builder_->createNetworkV2(_explicit_batch));
    _construct_network(graph, context);
    _setup_optimize(context); // config init
    builder_->setMaxBatchSize(context.max_batch_size);
    nvinfer1::IHostMemory* hm = builder_->buildSerializedNetwork(*network_, *config_);
    std::ofstream serialize_output_stream;
    std::string output_path = absl::StrReplaceAll(context.model_path, {{".onnx", ".plan"}});
    serialize_output_stream.open(output_path, std::ios::out|std::ios::binary);
    serialize_output_stream.write((const char*)hm->data(), hm->size());
    serialize_output_stream.close();
    ConvertContext __context;
    __context.model_path = output_path;
    de_serialize(graph, __context);
    return absl::OkStatus();
}

Status TensorRTEngine::de_serialize(Graph& graph, const ConvertContext& context) {
    std::fstream in;
	in.open(context.model_path, std::ios::binary | std::ios::in);
    if (!in.is_open()) {
        return absl::UnavailableError("TRT model open failed");
    }
    in.seekg(0, std::ios::end);
	int length = in.tellg();
	in.seekg(0, std::ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	in.read(data.get(), length);
	in.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime==nullptr) {
        return absl::InternalError("Create TRT IRuntime failed.");
    }
    bool didInitPlugins = initLibNvInferPlugins(&gLogger, "");
    MCHECK(didInitPlugins)<<"Init nvinfer plugins failed.";
    engine_.reset(runtime->deserializeCudaEngine(data.get(), length));
    if (engine_==nullptr) {
        return absl::InternalError("Create TRT ICudaEngine failed.");
    }
    context_.reset(engine_->createExecutionContext());
    if (context_==nullptr) {
        return absl::InternalError("Create TRT IExecutionContext failed.");
    }
    data.reset();
    runtime->destroy();
    cudaStreamCreate(&stream_);

    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine_->getBindingDataType(i);
        
        Tensor tensor(DeviceType::CUDA);
        std::vector<int64_t> shapes;
        for (int n = 0; n < dims.nbDims; ++n) {
            shapes.push_back(dims.d[n]);
        }
        IntArrayRef shape(shapes);
        tensor.set_shape(shape);
        tensor.set_name(engine_->getBindingName(i));
        switch (dtype) {
        case nvinfer1::DataType::kFLOAT :
            tensor.mutable_data<float>();
            break;
        case nvinfer1::DataType::kINT8 :
            tensor.mutable_data<int8_t>();
            break;
        case nvinfer1::DataType::kINT32 :
            tensor.mutable_data<int32_t>();
            break;
        default:
            MLOG(FATAL)<<"Unsupport dtype:"<<static_cast<int>(dtype);
        }
        if (engine_->bindingIsInput(i)) {
            itensors.push_back(tensor);
        } else {
            otensors.push_back(tensor);
        }
    }
    
    return absl::OkStatus();
}

Status TensorRTEngine::run(const ExecContext& context) {
    void* buffers[itensors.size()+otensors.size()];
    int i = 0;
    for (auto& tensor : itensors) {
        if (context.itensors.count(tensor.name()) == 0) {
            MLOG(FATAL)<<"Input name:"<<tensor.name()<<" do not in input tensors";
        } else {
            void*             input  = context.itensors.at(tensor.name()).input;
            const TypeMeta&   dtype  = context.itensors.at(tensor.name()).dtype;
            const DeviceType& device = context.itensors.at(tensor.name()).device;
            Shape shape{context.itensors.at(tensor.name()).shape};
            if (Device{device}.is_cpu()) {
                cudaMemcpyAsync(tensor.data(), input, shape.size()*dtype.itemsize(), cudaMemcpyHostToDevice, stream_);
                buffers[i] = tensor.data();
            } else {
                buffers[i] = input;
            }
        }
        i++;
    }
    for (auto& tensor : otensors) {
        buffers[i] = tensor.data();
        i++;
    }
    context_->enqueueV2(buffers, stream_, nullptr);
    cudaStreamSynchronize(stream_);
    return absl::OkStatus();
}

nvinfer1::ITensor* TensorRTEngine::_get_itensor(const std::string& iname) {
    if (nvtensor_map_.count(iname)) {
        return nvtensor_map_[iname];
    } else {
        MLOG(FATAL)<<"Can not find the input tensor:"<<iname<<" construct trt engine failed!!";
    }
}

Status TensorRTEngine::_construct_network(Graph& graph, const ConvertContext& context) {
    for (auto& it : context.ishapes) { // set network inputs
        std::string itname = it.first+input_prefix_;
        Shape shape(it.second);
        nvtensor_map_[itname] = _add_input(shape, itname, nvinfer1::DataType::kFLOAT);
    }
    
    for (auto& node : graph.order()) {
        std::cout<<"ddd:"<<node->name()<<std::endl;
        if (layer_make_map_.count(node->op_type())) {
            layer_make_map_[node->op_type()](this, *node, context);
        }
        if (node->outputs().size() == 0) { // set network outputs
            nvinfer1::ITensor *tensor = _get_itensor(node->name());
            tensor->setName(node->name().c_str());
            network_->markOutput(*tensor);
        }
    }
    return absl::OkStatus();
}

void TensorRTEngine::_setup_optimize(const ConvertContext& context) {
    config_.reset(builder_->createBuilderConfig());
    switch (context.mode) {
    case ModelMode::FP16 :
        config_->setFlag(nvinfer1::BuilderFlag::kFP16);
        break;
    case ModelMode::INT8 :
    case ModelMode::QATINT8 :
        config_->setFlag(nvinfer1::BuilderFlag::kINT8);
        break;
    default:
        MLOG(FATAL)<<"TRT unsupport mode type!!";
    }
    config_->setMaxWorkspaceSize(16_MiB);
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
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
