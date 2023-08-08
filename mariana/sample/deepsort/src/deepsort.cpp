#define _DLL_EXPORTS

#include "deepsort.h"

DeepSort::DeepSort() {
    this->maxBudget = 100;
    this->maxCosineDist = 0.2;
    init();
}


void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor = new FeatureTensor();
}

DeepSort::~DeepSort() {
    delete objTracker;
    delete featureExtractor;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;

    for (DetectBox i : dets) {
        DETECTBOX box(i.result.bbox.tl.x, i.result.bbox.tl.y,
                      i.result.bbox.w(), i.result.bbox.h());
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.result.score;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.result.cls_idx, i.result.score));
    }
    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b;
        b.result.bbox.tl.x = i(0);
        b.result.bbox.tl.y = i(1);
        b.result.bbox.br.x = i(2) + i(0);
        b.result.bbox.br.y = i(3) + i(1);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].result.cls_idx = c.cls;
        dets[i].result.score   = c.conf;
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));
        }
    }
}
