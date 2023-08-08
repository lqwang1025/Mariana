#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "model.hpp"
#include "datatype.h"
#include <mariana_api.h>

using std::vector;

class FeatureTensor {
public:
    FeatureTensor();
    ~FeatureTensor();

public:
    bool getRectsFeature(const cv::Mat& img, DETECTIONS& det);
    void infer(const cv::Mat& img);
    void stream2det(DETECTION_ROW& dbox);

private:
    const int featureDim = 512;
    mariana::ConvertContext ccontext;
    mariana::ExecContext econtext;
    mariana::Runtime* runtime = nullptr;
    int ih = 128;
    int iw = 64;
};

#endif
