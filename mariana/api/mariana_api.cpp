/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : mariana_api.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:09:48:55
 * Description:
 * 
 */

#include <mariana_api.h>
#include <marc/marc.h>
#include <core/utils/logging.h>
#include <structure/graph_exec.h>
#include <absl/strings/match.h>
#include <api/util/io.h>
#include <api/proto/mariana.pb.h>
#include <api/util/session_impl.h>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <core/cuda/device_attr.h>
#endif

namespace mariana {

Point2D::Point2D() {
    x = 0.f;
    y = 0.f;
}

Point2D::Point2D(float _x, float _y): x(_x), y(_y) {}

Point2D::~Point2D() {}

Rect::Rect() {}

Rect::~Rect() {}

float Rect::area() const {
    return w()*h();
}

float Rect::w() const {
    return (br.x-tl.x);
}

float Rect::h() const {
    return (br.y-tl.y);
}

Point2D Rect::cxy() const {
    return {(tl.x+br.x)/2, (tl.y+br.y)/2};
}

NNResult::NNResult() {
    score = 0.f;
    cls_idx = -1;
    class_name = "";
}

NNResult::~NNResult() {}

InferResult::InferResult() {}

InferResult::~InferResult() {}

MResult::MResult() {
    identification = "";
}

MResult::~MResult() {}

MTensor::MTensor() {
    input = nullptr;
    device = DeviceType::CPU;
    
}

MTensor::~MTensor() {}

void* MTensor::to_cpu(void* dst) const {
    int bsize = 1;
    for (auto& it : this->shape) {
        bsize*=it;
    }
    bsize*=dtype.itemsize();
    if (this->device == DeviceType::CPU) {
        if (dst == nullptr) {
            void* ptr = malloc(bsize);
            memcpy(ptr, this->input, bsize);
            return ptr;
        } else {
            memcpy(dst, this->input, bsize);
            return dst;
        }
        
    } else if (this->device == DeviceType::CUDA) {
#ifdef WITH_CUDA
        if (dst == nullptr) {
            void* ptr = malloc(bsize);
            cudaMemcpy(ptr, this->input, bsize, cudaMemcpyDeviceToHost);
            return ptr;
        } else {
            cudaMemcpy(dst, this->input, bsize, cudaMemcpyDeviceToHost);
            return dst;
        }
#else
        MLOG(FATAL)<<"Mariana compiling is not with CUDA!";
#endif
    }
}

ExecContext::ExecContext() {
    identification = "";
    info.pad_h = 0;
    info.pad_w = 0;
    info.scale = 1.f;
    anything = nullptr;
}

ExecContext::~ExecContext() {}

bool Runtime::build(const char* prototxt) {
    proto::ModelInfo model_info;
    bool success = read_proto_from_text_file(prototxt, &model_info);
    if (absl::EndsWith(model_info.model_path(), ".onnx")) {
        mariana::Graph* graph = mariana::parse(model_info);
        return true;
    }
    return false;
}

void Runtime::destory() {
    mariana::Graph* graph = static_cast<mariana::Graph*>(handle_);
    delete graph;
}

Runtime Runtime::create_from(const char* prototxt) {
    proto::ModelInfo model_info;
    bool success = read_proto_from_text_file(prototxt, &model_info);
    Runtime runtime;
    if (false == success) {
        MLOG(ERROR)<<"By opening protobuf file:"<<prototxt<<" to construct runtime failed.";
        return runtime;
    }
    mariana::Graph* graph = mariana::parse(model_info);
    
    if (graph->engine()) {
        for (auto &it : graph->engine()->itensors) {
            runtime.input_names.push_back(it.name());
        }
        for (auto &it : graph->engine()->otensors) {
            runtime.output_names.push_back(it.name());
        }
    }  else {
        MCHECK(false);
    }
    runtime.handle_ = graph;
    return runtime;
}

MResult Runtime::run_with(ExecContext& econtext) {
    mariana::GraphExec ge;
    mariana::Graph* graph = static_cast<mariana::Graph*>(handle_);
    ge.run(*graph, econtext);
    return ge.results;
}

ScheduleConfig::ScheduleConfig() : num_thread(1), max_ibuffer_size(1) {}

ScheduleConfig::~ScheduleConfig() {}

Session::Session() {
    handle_ = new SessionImpl{};
}

Session::~Session() {
    SessionImpl* session = static_cast<SessionImpl*>(handle_);
    delete session;
}

void Session::add_channel(const char* channel_id, void* do_not_change) {
    RuntimeImpl* runtime = static_cast<RuntimeImpl*>(do_not_change);
    runtime->add_obuffer(std::string(channel_id));
}

void* Session::add_section(const char* prototxt, const ScheduleConfig& config) {
    SessionImpl* session = static_cast<SessionImpl*>(handle_);
    RuntimeImpl* runtime_impl = session->add_runtime(prototxt, config);
    return runtime_impl;
}

void Session::push_with(ExecContext& context, void* do_not_change) {
    RuntimeImpl* runtime = static_cast<RuntimeImpl*>(do_not_change);
    runtime->push(context);
}

MResult Session::pull(ExecContext& context, void* do_not_change) {
    RuntimeImpl* runtime = static_cast<RuntimeImpl*>(do_not_change);
    return runtime->pull(context);
}

std::vector<std::string> Session::get_onames_from(void* do_not_change) {
    RuntimeImpl* runtime = static_cast<RuntimeImpl*>(do_not_change);
    return runtime->inames();
}

std::vector<std::string> Session::get_inames_from(void* do_not_change) {
    RuntimeImpl* runtime = static_cast<RuntimeImpl*>(do_not_change);
    return runtime->onames();
}

} // namespace mariana
