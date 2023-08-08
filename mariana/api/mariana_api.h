/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : mariana_api.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:08:34:05
 * Description:
 *
 */

#ifndef __MARIANA_API_H__
#define __MARIANA_API_H__

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <typemeta.h>

namespace mariana {

enum class Backend : int8_t {
    UNINIT = -1,
    RKNN   = 0,
    TRT    = 1,
};

enum class ModelMode : int8_t {
    UNINIT  = -1,
    FP16    = 0,
    FP32    = 1,
    INT8    = 2,
    QATINT8 = 3,
};

enum class ProcessorCategory : int8_t {
    UNINIT  = -1,
    YOLOV8_POST_ONE_OUTPUT = 0,
    YOLOV8_POST_THREE_OUTPUT  = 1,
};

enum class DeviceType : int8_t { // DO NOT CHANGE THE NUMBERS OF BELLOW.
    UNINIT=0,
    CPU=0,
    CUDA=1,
    FPGA=2,
    // If you add other devices
    //  - Change the implementations of DeviceTypeName and isValidDeviceType
    //    in device_ype.cpp
    //  - Change the number below
    COMPILE_TIME_MAX_DEVICE_TYPES=3
};

struct Point2D {
    Point2D() {
        x = 0.f;
        y = 0.f;
    }
    Point2D(float _x, float _y): x(_x), y(_y) {}
    ~Point2D() {}
    float x;
    float y;
};

struct Rect {
    Rect() {}
    ~Rect() {}
    float area() const {
        return w()*h();
    }
    float w() const {
        return (br.x-tl.x);
    }
    float h() const {
        return (br.y-tl.y);
    }
    Point2D cxy() const {
        return {(tl.x+br.x)/2, (tl.y+br.y)/2};
    }
    Point2D tl;
    Point2D br;
};

struct MResult {
    MResult() {
        score = 0.f;
        batch_idx = -1;
        cls_idx = -1;
        class_name = "";
    }
    ~MResult() {}
    float score;
    int32_t batch_idx;
    int32_t cls_idx;
    std::string class_name;
    std::vector<Point2D> points;
    Rect bbox;
};

struct ConvertContext {
    ConvertContext() {
        model_path = "";
        from_scratch = false;
        back_end = Backend::UNINIT;
        max_batch_size = 1;
        mode = ModelMode::FP16;
        procategory = ProcessorCategory::UNINIT;
        iou_thresh = 0.f;
        conf_thresh = 0.f;
        grid_offset = 0.f;
    }
    ~ConvertContext() {}
    std::unordered_map<std::string, std::vector<int32_t>> ishapes;
    std::string model_path;
    bool from_scratch;
    Backend back_end;
    int max_batch_size;
    ModelMode mode;
    ProcessorCategory procategory;
    float iou_thresh;
    float conf_thresh;
    std::vector<int> grids;
    std::vector<int> strides;
    float grid_offset;
    std::vector<std::string> labels;
};

struct MTensor { // It hold external data pointer only, is not responsible for freeing memory.
    MTensor() {
        input = nullptr;
        device = DeviceType::CPU;
    }
    ~MTensor() {}
    std::vector<int32_t> shape;
    void* input;
    TypeMeta dtype;
    DeviceType device;
};

struct ExecContext {
    ExecContext() {
        pad_h = 0;
        pad_w = 0;
        scale = 1.f;
    }
    ~ExecContext() {   
    }
    std::unordered_map<std::string, MTensor> itensors;
    std::vector<MTensor> otensors;
    int pad_h;
    int pad_w;
    float scale;
};

struct Runtime {
    Runtime(const ConvertContext& ccontext);
    ~Runtime();
    std::vector<MResult> run_with(ExecContext& econtext);
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
private:
    void* handle_ = nullptr;
};

} // namespace mariana

#endif /* __MARIANA_API_H__ */

