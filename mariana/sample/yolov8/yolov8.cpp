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
#include <sys/time.h>

using result_list = std::vector<mariana::MResult>;

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static float iou_of(const mariana::Rect& a, const mariana::Rect& b) {
	float max_x = std::max(a.tl.x, b.tl.x);
	float max_y = std::max(a.tl.y, b.tl.y);
	float min_x = std::min(a.br.x, b.br.x);
	float min_y = std::min(a.br.y, b.br.y);

	float h = std::max(0.f, min_y-max_y);
	float w = std::max(0.f, min_x-max_x);
	float inter_area = h*w;
	float union_area = a.area() + b.area() - inter_area;
	return inter_area/union_area;
}

static void nms(result_list& dets, float iou_thresh) {
	result_list results;
	while (!dets.empty()) {
		std::sort(dets.begin(), dets.end(), [&](mariana::MResult& a, mariana::MResult& b) {return a.score < b.score;});
		results.push_back(dets[0]);
		for (auto it = dets.begin()+1; it != dets.end(); ++it) {
			float iou = iou_of(dets[0].bbox, it->bbox);
			if (iou > iou_thresh) {
				it = dets.erase(it);
				it--;
			}
		}
		dets.erase(dets.begin());
	}
	dets = results;
}


result_list yolov8_post(mariana::ExecContext& context, const mariana::ConvertContext& option) {
    result_list results;
    mariana::MTensor tensor = context.otensors[0];
    std::vector<int> shape  = tensor.shape;
    int nb = shape[0];
    int ng = shape[1];
    int nc = shape[2];
    for (int n = 0; n < nb; ++n) {
        result_list __results;
        int nstride = n*ng*nc;
        for (int g = 0; g < ng; ++g) {
            int gstride = g*nc;
            float max = -FLT_MAX;
            int index = -1;
            for (int c = 4; c < nc; ++c) {
                float score = static_cast<float*>(tensor.input)[nstride+gstride+c];
                if (max < score) {
                    max = score;
                    index = c-4;
                }
            }
            if (max < option.conf_thresh) continue;

            float cx = static_cast<float*>(tensor.input)[nstride+gstride+0];
			float cy = static_cast<float*>(tensor.input)[nstride+gstride+1];
			float w	 = static_cast<float*>(tensor.input)[nstride+gstride+2];
			float h	 = static_cast<float*>(tensor.input)[nstride+gstride+3];
            mariana::MResult result;
            result.batch_idx  = n;
            result.cls_idx    = index;
            result.class_name = option.labels[index];
            result.score      = max;
            result.bbox.tl.x  = (cx - w/2 - context.pad_w)/context.scale;
            result.bbox.tl.y  = (cy - h/2 - context.pad_h)/context.scale;
            result.bbox.br.x  = (cx + w/2 - context.pad_w)/context.scale;
            result.bbox.br.y  = (cy + h/2 - context.pad_h)/context.scale;
            __results.push_back(result);
        }
        nms(__results, option.iou_thresh);
        results.insert(results.end(), __results.begin(), __results.end());
    }
    return results;    
}

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
    const int imgsz = 800;
    mariana::ConvertContext ccontext;
    ccontext.ishapes.insert({"Conv_3", {1, 3, imgsz, imgsz}});
    ccontext.model_path = "../models/yolov8x_silu.plan";
    ccontext.back_end   = mariana::Backend::TRT;
    ccontext.procategory = mariana::ProcessorCategory::YOLOV8_POST_ONE_OUTPUT;
    ccontext.iou_thresh = 0.6f;
    ccontext.conf_thresh = 0.25f;
    //ccontext.from_scratch = true;
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
    cv::Mat resized = letterbox(src, imgsz, imgsz, econtext);
    cv::Mat input = cv::dnn::blobFromImage(resized, 0.00392157d/*scale*/, cv::Size(), cv::Scalar(), /*swapRB*/false);
    mariana::MTensor tensor;
    tensor.shape  = {1, 3, imgsz, imgsz};
    tensor.dtype  = mariana::TypeMeta::make<float>();
    tensor.input  = input.data;
    tensor.device = mariana::DeviceType::CPU;
    econtext.itensors.insert({"images", tensor});
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    runtime.run_with(econtext);
    std::vector<mariana::MResult> results =  yolov8_post(econtext, ccontext);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    
    gettimeofday(&start_time, NULL);
    for (int i  = 0; i < 10; ++i ) {
        runtime.run_with(econtext);
        std::vector<mariana::MResult> results =  yolov8_post(econtext, ccontext);
    }
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 10000);
    
    for (auto& it : results) {
		cv::rectangle(src, cv::Rect(it.bbox.tl.x, it.bbox.tl.y, it.bbox.w(), it.bbox.h()), cv::Scalar(0, 0, 255), 4);
        cv::putText(src, it.class_name, cv::Point(it.bbox.tl.x, it.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(114,114,114), 2, 8, false);
	}
    cv::imwrite("res.jpg", src);
    return 0;
}
