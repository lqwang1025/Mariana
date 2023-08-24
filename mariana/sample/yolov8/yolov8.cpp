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
		std::sort(dets.begin(), dets.end(), [&](mariana::MResult& a, mariana::MResult& b) {return a.score > b.score;});
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
    float* buffer = nullptr;
    if (tensor.device == mariana::DeviceType::CPU) {
        buffer = static_cast<float*>(tensor.input);
    } else if (tensor.device == mariana::DeviceType::CUDA) {
        buffer = static_cast<float*>(tensor.to_cpu());
    } else {
        abort();
    }
    
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
                float score = buffer[nstride+gstride+c];
                if (max < score) {
                    max = score;
                    index = c-4;
                }
            }
            if (max < option.conf_thresh) continue;
            
            float cx = buffer[nstride+gstride+0];
			float cy = buffer[nstride+gstride+1];
			float w	 = buffer[nstride+gstride+2];
			float h	 = buffer[nstride+gstride+3];
            mariana::MResult result;
            result.batch_idx  = n;
            result.cls_idx    = index;
            result.class_name = option.labels[index];
            result.score      = max;
            result.bbox.tl.x  = (cx - w/2 - context.pad_w)/context.scale+index*600;
            result.bbox.tl.y  = (cy - h/2 - context.pad_h)/context.scale+index*600;
            result.bbox.br.x  = (cx + w/2 - context.pad_w)/context.scale+index*600;
            result.bbox.br.y  = (cy + h/2 - context.pad_h)/context.scale+index*600;
            __results.push_back(result);
        }
        nms(__results, option.iou_thresh);
        for (auto& it : __results) {
            it.bbox.tl.x -= it.cls_idx*600;
            it.bbox.tl.y -= it.cls_idx*600;
            it.bbox.br.x -= it.cls_idx*600;
            it.bbox.br.y -= it.cls_idx*600;
            results.push_back(it);
        }
    }
    if (buffer) free(buffer);
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

class Classification {
public:
    Classification() {
        mariana::ConvertContext ccontext;
        ccontext.back_end   = mariana::Backend::TRT;
        ccontext.model_path = "../models/cls.plan";
        runtime = new mariana::Runtime(ccontext);
    }

    ~Classification() {
        delete runtime;
    }
    
    void run(const cv::Mat& src) {
        mariana::ExecContext econtext;
        cv::Mat input = cv::dnn::blobFromImage(src, 0.00392157d/*scale*/, cv::Size(imgsz, imgsz), cv::Scalar(), /*swapRB*/true);
        mariana::MTensor tensor;
        tensor.shape  = {1, 3, imgsz, imgsz};
        tensor.dtype  = mariana::TypeMeta::make<float>();
        tensor.input  = input.data;
        tensor.device = mariana::DeviceType::CPU;
        econtext.itensors.clear();
        econtext.itensors.insert({"x.1", tensor});
        runtime->run_with(econtext);
        mariana::MTensor& otensor = econtext.otensors[0];
        float max = -FLT_MAX;
        int index = -1;

        float* buffer = nullptr;
        if (otensor.device == mariana::DeviceType::CPU) {
            buffer = static_cast<float*>(otensor.input);
        } else if (otensor.device == mariana::DeviceType::CUDA) {
            buffer = static_cast<float*>(otensor.to_cpu(outptr));
        } else {
            abort();
        }
        
        for (int i = 0; i < 7; ++i) {
            float score = buffer[i];
            if (max < score) {
                max = score;
                index = i;
            }
        }
        if (labels[index] == "车辆") {
            cv::imwrite("car_"+std::to_string(count++)+".jpg", src);
        }
        std::cout<<"cls res:"<<labels[index]<<" "<<max<<std::endl;
    }
    int count = 0;
    float outptr[7] = {0};
    mariana::Runtime* runtime = nullptr;
    const int imgsz = 224;
    std::vector<std::string> labels = {"背景", "行人", "施工机械", "车辆", "火焰", "低照度膨胀火", "烟雾"};

};

int main(int argv, const char* argc[]) {
    const int imgsz = 800;
    mariana::ConvertContext ccontext;
    ccontext.ishapes.insert({"Conv_3", {1, 3, imgsz, imgsz}});
    ccontext.model_path = "../models/yolov8x_relu.plan";
    ccontext.back_end   = mariana::Backend::TRT;
    ccontext.procategory = mariana::ProcessorCategory::YOLOV8_POST_ONE_OUTPUT;
    ccontext.iou_thresh = 0.6f;
    ccontext.conf_thresh = 0.3f;
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

    cv::VideoCapture vc;
    vc.open("job_leave.mp4");
    if (!vc.isOpened()) {
		printf("could not read this video file...\n");
		return -1;
	}
    cv::Mat frame;
    cv::Size S = cv::Size((int)vc.get(cv::CAP_PROP_FRAME_WIDTH),
                          (int)vc.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = vc.get(cv::CAP_PROP_FPS);
    //cv::VideoWriter writer("res.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps, S, true);
    int count = 0;

    Classification cls;
    
    while(vc.isOpened()) {
        bool ret = vc.grab();
        count++;
        if (!ret) continue;
        //if ( count%20 != 0) continue;
        ret = vc.retrieve(frame);
        std::cout<<"ss:"<<ret<<" "<<frame.size()<<std::endl;
        cv::Mat resized = letterbox(frame, imgsz, imgsz, econtext);
        cv::Mat input = cv::dnn::blobFromImage(resized, 0.00392157d/*scale*/, cv::Size(), cv::Scalar(), /*swapRB*/false);
        mariana::MTensor tensor;
        tensor.shape  = {1, 3, imgsz, imgsz};
        tensor.dtype  = mariana::TypeMeta::make<float>();
        tensor.input  = input.data;
        tensor.device = mariana::DeviceType::CPU;
        econtext.itensors.clear();
        econtext.itensors.insert({"images", tensor});
        runtime.run_with(econtext);
        // std::cout<<results.size()<<std::endl;
        std::vector<mariana::MResult> results = yolov8_post(econtext, ccontext);
        int pc = 0;
        for (auto& it : results) {
            //std::cout<<"detect:"<<it.class_name<<" "<<it.score<<std::endl;
            if (it.class_name != "person") continue;
            cv::Rect roi(it.bbox.tl.x, it.bbox.tl.y, it.bbox.w(), it.bbox.h());
            cv::Mat person = frame(roi);
            cls.run(person);
            //cv::imwrite("person"+std::to_string(count++)+std::to_string(pc++)+".jpg", person);
            cv::rectangle(frame, roi, cv::Scalar(0, 0, 255), 4);
            cv::putText(frame, it.class_name+":"+std::to_string(it.score), cv::Point(it.bbox.tl.x, it.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2, 8, false);
        }
        //cv::imwrite("test"+std::to_string(count)+".jpg", frame);
        //writer.write(frame);
    }
    // while (vc.read(frame)) {
	// 	// imshow("camera-demo", frame);
	// 	// writer.write(frame);
	// 	// char c = waitKey(50);
	// 	// if (c == 27) {
	// 	// 	break;
	// 	// }
    //     cv::Mat resized = letterbox(frame, imgsz, imgsz, econtext);
    //     cv::Mat input = cv::dnn::blobFromImage(resized, 0.00392157d/*scale*/, cv::Size(), cv::Scalar(), /*swapRB*/false);
    //     mariana::MTensor tensor;
    //     tensor.shape  = {1, 3, imgsz, imgsz};
    //     tensor.dtype  = mariana::TypeMeta::make<float>();
    //     tensor.input  = input.data;
    //     tensor.device = mariana::DeviceType::CPU;
    //     econtext.itensors.clear();
    //     econtext.itensors.insert({"images", tensor});
    //     std::vector<mariana::MResult> results = runtime.run_with(econtext);
    //     // std::cout<<results.size()<<std::endl;
    //      // =  yolov8_post(econtext, ccontext);
    //     for (auto& it : results) {
    //         //std::cout<<"detect:"<<it.class_name<<" "<<it.score<<std::endl;
    //         // if (it.score <= 0.522f) continue;
    //         cv::rectangle(frame, cv::Rect(it.bbox.tl.x, it.bbox.tl.y, it.bbox.w(), it.bbox.h()), cv::Scalar(0, 0, 255), 4);
    //         cv::putText(frame, it.class_name+":"+std::to_string(it.score), cv::Point(it.bbox.tl.x, it.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2, 8, false);
    //     }
    //     writer.write(frame);
    //     //cv::imwrite("res/test"+std::to_string(count++)+".jpg", frame);
        
	// }
    vc.release();
//    writer.release();
    // cv::Mat src = cv::imread("./20230620_145856_火灾_74体育馆西门外球机_sample.jpg");
    // cv::Mat resized = letterbox(src, imgsz, imgsz, econtext);
    // cv::Mat input = cv::dnn::blobFromImage(resized, 0.00392157d/*scale*/, cv::Size(), cv::Scalar(), /*swapRB*/false);
    // mariana::MTensor tensor;
    // tensor.shape  = {1, 3, imgsz, imgsz};
    // tensor.dtype  = mariana::TypeMeta::make<float>();
    // tensor.input  = input.data;
    // tensor.device = mariana::DeviceType::CPU;
    // econtext.itensors.insert({"images", tensor});
    // struct timeval start_time, stop_time;
    // gettimeofday(&start_time, NULL);
    // runtime.run_with(econtext);
    // std::vector<mariana::MResult> results =  yolov8_post(econtext, ccontext);
    // gettimeofday(&stop_time, NULL);
    // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    
    // gettimeofday(&start_time, NULL);
    // for (int i  = 0; i < 100; ++i ) {
    //     std::vector<mariana::MResult> results = runtime.run_with(econtext);
    //     //std::vector<mariana::MResult> results =  yolov8_post(econtext, ccontext);
    // }
    // gettimeofday(&stop_time, NULL);
    // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 100000);
    
    // for (auto& it : results) {
    //     std::cout<<"detect:"<<it.class_name<<" "<<it.score<<std::endl;
    //     // if (it.score <= 0.522f) continue;
	// 	cv::rectangle(src, cv::Rect(it.bbox.tl.x, it.bbox.tl.y, it.bbox.w(), it.bbox.h()), cv::Scalar(0, 0, 255), 4);
    //     cv::putText(src, it.class_name, cv::Point(it.bbox.tl.x, it.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2, 8, false);
	// }
    // cv::imwrite("res.jpg", src);
    return 0;
}
