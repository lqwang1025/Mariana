/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : detect.h
 * Authors    : lqwang@pandora
 * Create Time: 2023-08-06:09:16:35
 * Description:
 *
 */

#ifndef __DETECT_H__
#define __DETECT_H__

#include <opencv2/opencv.hpp>
#include <mariana_api.h>

class Yolov8Detect {
public:
    Yolov8Detect() {
        ccontext.model_path = "./model/yolov8s_80.rknn";
        ccontext.back_end   = mariana::Backend::RKNN;
        ccontext.procategory = mariana::ProcessorCategory::YOLOV8_POST_THREE_OUTPUT;
        ccontext.iou_thresh = 0.6f;
        ccontext.conf_thresh = 0.25f;
        ccontext.grids = {80, 40, 20};
        ccontext.strides = {8, 16, 32};
        ccontext.grid_offset = 0.5f;
        ccontext.labels = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
        runtime = new mariana::Runtime(ccontext);
    }
    ~Yolov8Detect() {
        if (runtime)
            delete runtime;
        runtime = nullptr;
    }
    cv::Mat letterbox(const cv::Mat &src, int h, int w);
    std::vector<mariana::MResult> infer(const cv::Mat& src);
    mariana::ConvertContext ccontext;
    mariana::ExecContext econtext;
    mariana::Runtime* runtime = nullptr;
    bool rgb = true;
    int ih = 640;
    int iw = 640;
};

#endif /* __DETECT_H__ */

