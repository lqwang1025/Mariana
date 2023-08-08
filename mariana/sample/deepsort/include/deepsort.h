#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "featuretensor.h"
#include "tracker.h"
#include "datatype.h"
#include <vector>

using std::vector;

class DeepSort {
public:
    DeepSort();
    ~DeepSort();

public:
    void sort(cv::Mat& frame, vector<DetectBox>& dets);
     
private:
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);
    void init();

private:
    std::string enginePath;
    int batchSize;
    int featureDim;
    cv::Size imgShape;
    int maxBudget;
    float maxCosineDist;

private:
    vector<RESULT_DATA> result;
    vector<std::pair<CLSCONF, DETECTBOX>> results;
    tracker* objTracker = nullptr;
    FeatureTensor* featureExtractor = nullptr;
};

#endif  //deepsort.h
