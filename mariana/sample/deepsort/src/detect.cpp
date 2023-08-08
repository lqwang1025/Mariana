/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : detect.cpp
 * Authors    : lqwang@pandora
 * Create Time: 2023-08-06:09:20:26
 * Description:
 * 
 */

#include <detect.h>
#include <sys/time.h>
#include <cstdio>

cv::Mat Yolov8Detect::letterbox(const cv::Mat &src, int h, int w) {
    int		in_w	 = src.cols;	// width
    int		in_h	 = src.rows;	// height
    int		tar_w	 = w;
    int		tar_h	 = h;
    float	r		 = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int		inside_w = round(in_w * r);
    int		inside_h = round(in_h * r);
    int		padd_w	 = tar_w - inside_w;
    int		padd_h	 = tar_h - inside_h;
    
    econtext.scale = r;
    cv::Mat resize_img;

    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
	econtext.pad_h = padd_h;
	econtext.pad_w = padd_w;
	
    int top	   = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left   = int(round(padd_w - 0.1));
    int right  = int(round(padd_w + 0.1));
    cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
	if (rgb) {
		cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
	}
    return resize_img;
}

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

std::vector<mariana::MResult> Yolov8Detect::infer(const cv::Mat& src) {
    struct timeval start_time, stop_time;
    cv::Mat resized = letterbox(src, ih, iw);
    mariana::MTensor tensor;
    tensor.shape  = {1, 3, ih, iw};
    tensor.dtype  = mariana::TypeMeta::make<uint8_t>();
    tensor.input  = resized.data;
    econtext.itensors.insert({runtime->input_names[0], tensor});
    gettimeofday(&start_time, NULL);
    std::vector<mariana::MResult> results = runtime->run_with(econtext);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    return results;
}
