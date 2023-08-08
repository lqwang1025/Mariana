#include "featuretensor.h"
#include <fstream>

FeatureTensor::FeatureTensor() {
    ccontext.model_path = "./model/deepsort.rknn";
    ccontext.back_end   = mariana::Backend::RKNN;
    runtime = new mariana::Runtime(ccontext);
}

FeatureTensor::~FeatureTensor() {
    if (runtime)
        delete runtime;
    runtime = nullptr;
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));
        cv::Mat tempMat = img(rect).clone();
        infer(tempMat);
        stream2det(dbox);
    }
    return true;
}

void FeatureTensor::infer(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(iw, ih));
    mariana::MTensor tensor;
    tensor.shape  = {1, 3, ih, iw};
    tensor.dtype  = mariana::TypeMeta::make<uint8_t>();
    tensor.input  = resized.data;
    econtext.itensors.insert({runtime->input_names[0], tensor});
    runtime->run_with(econtext);
}

void FeatureTensor::stream2det(DETECTION_ROW& dbox) {
    int i = 0;
    for (int j = 0; j < featureDim; ++j) {
        float data = static_cast<float*>(econtext.otensors[0].input)[j];
        dbox.feature[j] = data;
    }
}
