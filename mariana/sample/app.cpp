/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : app.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:08:57:10
 * Description:
 * 
 */

#include <iostream>
#include <mariana_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


cv::Mat letterbox(cv::Mat &src, int h, int w, mariana::ExecContext& context, bool rgb=true) {

    int		in_w	 = src.cols;	// width
    int		in_h	 = src.rows;	// height
    int		tar_w	 = w;
    int		tar_h	 = h;
    float	r		 = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int		inside_w = round(in_w * r);
    int		inside_h = round(in_h * r);
    int		padd_w	 = tar_w - inside_w;
    int		padd_h	 = tar_h - inside_h;
	
    cv::Mat resize_img;

    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;

    context.pad_h = padd_h;
    context.pad_w = padd_w;
    context.scale = r;
    
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

int main(int argv, const char* argc[]) {
    mariana::ConvertContext ccontext;
    //ccontext.ishapes.insert({"Conv_0", {1, 3, 640, 640}});
    ccontext.model_path = "../models/yolov8l_relu.plan";
    ccontext.back_end   = mariana::Backend::TRT;
    ccontext.procategory = mariana::ProcessorCategory::YOLOV8_POST_ONE_OUTPUT;
    ccontext.iou_thresh = 0.6f;
    ccontext.conf_thresh = 0.25f;
    ccontext.labels = {"flame",
                       "flame_fu",
                       "smoke",
                       "smoke_dif",
                       "person",
                       "car",
                       "truck",
                       "bus",
                       "electric_tricycle",
                       "excavator",
                       "bicycle",
                       "motorcycle",
                       "fire hydrant",
                       "stop sign",
                       "tv","cone","light"};
    mariana::Runtime runtime(ccontext);
    
    mariana::ExecContext econtext;
    cv::Mat src = cv::imread("./165.jpg");
    cv::Mat resized = letterbox(src, 640, 640, econtext);
    cv::imwrite("input.jpg", resized);
    cv::Mat input = cv::dnn::blobFromImage(resized, 0.00392157d/*scale*/, cv::Size(), cv::Scalar(), /*swapRB*/false);
    std::cout<<"d:"<<input.size<<std::endl;
    mariana::MTensor tensor;
    tensor.shape  = {1, 3, 640, 640};
    tensor.dtype  = mariana::TypeMeta::make<float>();
    tensor.input  = input.data;
    tensor.device = mariana::DeviceType::CPU;
    econtext.itensors.insert({"images", tensor});
    std::vector<mariana::MResult> results = runtime.run_with(econtext);
    for (auto& it : results) {
		cv::rectangle(src, cv::Rect(it.bbox.tl.x, it.bbox.tl.y, it.bbox.w(), it.bbox.h()), cv::Scalar(0, 0, 255), 4);
        cv::putText(src, it.class_name, cv::Point(it.bbox.tl.x, it.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(114,114,114), 2, 8, false);
	}
    cv::imwrite("res.jpg", src);
    return 0;
}
