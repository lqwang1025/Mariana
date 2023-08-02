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
    std::vector<std::string> labels;
};

struct Yolov8OnePostProcessor : public Processor {
    Yolov8OnePostProcessor(const ConvertContext& context) {
        option.iou_thresh  = context.iou_thresh;
        option.conf_thresh = context.conf_thresh;
        option.labels      = context.labels;
    }
    ~Yolov8OnePostProcessor() {}
    Yolov8PostOption option;
    result_list work(tensor_list&& inputs, ExecContext& context) override;
};

struct Yolov8ThreePostProcessor : public Processor {
    Yolov8ThreePostProcessor(const ConvertContext& context) {
        option.iou_thresh  = context.iou_thresh;
        option.conf_thresh = context.conf_thresh;
        option.grids       = context.grids;
        option.strides     = context.strides;
        option.labels      = context.labels;
    }
    ~Yolov8ThreePostProcessor() {}
    Yolov8PostOption option;
    result_list work(tensor_list&& inputs, ExecContext& context) override;
};


} // namespace mariana

#endif /* __STRUCTURE_PROCESSORS_YOLOV8_POST_H__ */

