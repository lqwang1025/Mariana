/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : main.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-26:16:43:23
 * Description:
 * 
 */

#include <detect.h>
#include <deepsort.h>
#include <cstdio>
#include <string>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static void run(const std::string& video_path) {
    Yolov8Detect detect;
    DeepSort tracker;
    int i = 1;
    struct timeval start_time, stop_time;
    for (; i < 303; ++i) {
        cv::Mat src = cv::imread(video_path+std::to_string(i)+".jpg");
        // gettimeofday(&start_time, NULL);
        std::vector<mariana::MResult> results = detect.infer(src);
        // gettimeofday(&stop_time, NULL);
        // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
        std::vector<DetectBox> det_boxes;
        for (auto& it : results) {
            if (it.class_name == "person") {
                DetectBox db;
                db.result = it;
                det_boxes.push_back(db);
            }
        }
        if (det_boxes.empty()) continue;
        tracker.sort(src, det_boxes);
        
        for (auto& it : det_boxes) {
            cv::rectangle(src, cv::Rect(it.result.bbox.tl.x, it.result.bbox.tl.y, it.result.bbox.w(), it.result.bbox.h()), cv::Scalar(0, 0, 255), 4);
            cv::putText(src, detect.ccontext.labels[it.result.cls_idx], cv::Point(it.result.bbox.tl.x, it.result.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(114,114,114), 2, 8, false);
            cv::putText(src, std::to_string(int(it.trackID)), cv::Point(it.result.bbox.tl.x+it.result.bbox.w(), it.result.bbox.tl.y+it.result.bbox.h()), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2, 8, false);
        }
        cv::imwrite(std::to_string(i)+".jpg", src);
    }
    std::cout<<"done"<<std::endl;
}


int main(int argc, char** argv) {
    run("./input/image");
	return 0;
}
