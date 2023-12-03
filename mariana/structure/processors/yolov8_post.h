/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/processors/yolov8_post.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:13:08:35
 * Description:
 *
 */

#ifndef __STRUCTURE_PROCESSORS_YOLOV8_POST_H__
#define __STRUCTURE_PROCESSORS_YOLOV8_POST_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/processor.h>
#include <structure/func_option.h>

namespace mariana {

struct Yolov8PostOption : public BaseOption {
    Yolov8PostOption() {
        grids.clear();
        strides.clear();
    }
    ~Yolov8PostOption() {}
    float iou_thresh = 0.f;
    float conf_thresh = 0.f;
    std::vector<int> grids;
    std::vector<int> strides;
    float grid_offset = 0.f;
    std::vector<std::string> labels;
};

struct Yolov8OnePostProcessor : public Processor {
    Yolov8OnePostProcessor(const proto::ModelInfo& model_info) {
        option.iou_thresh  = model_info.iou_thresh();
        option.conf_thresh = model_info.conf_thresh();
        option.labels.reserve(model_info.labels_size());
        for (size_t i = 0; i < model_info.labels_size(); ++i) {
            option.labels.push_back(model_info.labels(i));
        }
    }
    ~Yolov8OnePostProcessor() {}
    Yolov8PostOption option;
    MResult work(tensor_list&& inputs, ExecContext& context) override;
};

struct Yolov8ThreePostProcessor : public Processor {
    Yolov8ThreePostProcessor(const proto::ModelInfo& model_info) {
        option.iou_thresh  = model_info.iou_thresh();
        option.conf_thresh = model_info.conf_thresh();
        option.grid_offset = model_info.grid_offset();
        
        option.labels.reserve(model_info.labels_size());
        for (size_t i = 0; i < model_info.labels_size(); ++i) {
            option.labels.push_back(model_info.labels(i));
        }

        option.grids.reserve(model_info.grids_size());
        for (size_t i = 0; i < model_info.grids_size(); ++i) {
            option.grids.push_back(model_info.grids(i));
        }

        option.strides.reserve(model_info.strides_size());
        for (size_t i = 0; i < model_info.strides_size(); ++i) {
            option.strides.push_back(model_info.strides(i));
        }
        
    }
    ~Yolov8ThreePostProcessor() {}
    Yolov8PostOption option;
    MResult work(tensor_list&& inputs, ExecContext& context) override;
};


} // namespace mariana

#endif /* __STRUCTURE_PROCESSORS_YOLOV8_POST_H__ */

