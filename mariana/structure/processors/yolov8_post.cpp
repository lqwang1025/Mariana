/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/processors/yolov8_post.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:13:09:39
 * Description:
 * 
 */

#include <iostream>
#include <cfloat>
#include <structure/processors/yolov8_post.h>

namespace mariana {

static float sigmoid(float x) {
    return 1 / (1+expf(-x));
}

static std::vector<float> softmax(float* data, int n) {
    std::vector<float> res;
    res.reserve(n);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        res.push_back(expf(data[i]));
        sum += res[i];
    }
    for (int i = 0; i < n; ++i) {
        res[i] /= sum;
    }
    
    return res;
}

static float iou_of(const Rect& a, const Rect& b) {
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
		std::sort(dets.begin(), dets.end(), [&](MResult& a, MResult& b) {return a.score < b.score;});
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

result_list Yolov8OnePostProcessor::work(tensor_list&& inputs, ExecContext& context) {
    // The shape like : 1x8400x(4+cls_number)
    Tensor tensor = inputs[0].cpu();
    result_list results;
    Shape shape = tensor.shape();
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
                float score = tensor.data_ptr_impl<float>()[nstride+gstride+c];
                if (max < score) {
                    max = score;
                    index = c-4;
                }
            }
            if (max < option.conf_thresh) continue;

            float cx = tensor.data_ptr_impl<float>()[nstride+gstride+0];
            float cy = tensor.data_ptr_impl<float>()[nstride+gstride+1];
            float w  = tensor.data_ptr_impl<float>()[nstride+gstride+2];
            float h  = tensor.data_ptr_impl<float>()[nstride+gstride+3];
            MResult result;
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

result_list Yolov8ThreePostProcessor::work(tensor_list&& inputs, ExecContext& context) {
    /*
	 * output shape like: 1x8400x81 (81=4*16+nc)
	 * 8400 = 80x80+40x40+20x20
	 */
    result_list results;
    float* buffer = static_cast<float*>(inputs[0].data());
    Shape shape1 = inputs[0].shape(); // [1, 8400, 81]
    int nb = shape1[0];
    int nc = shape1[2]-4*16;
    for (int n = 0; n < nb; ++n) {
        result_list __results;
        for (size_t i = 0; i < option.grids.size(); ++i) {
            for (int h = 0; h < option.grids[i]; ++h) {
                int h_offset = h*option.grids[i]*(64+nc);
                for (int w = 0; w < option.grids[i]; ++w) {
                    int w_offset = w*(64+nc);
                    float max = -FLT_MAX;
                    int index = -1;
                    for (int c = 0; c < nc; ++c) {
                        float score = sigmoid(buffer[h_offset+w_offset+64+c]);
                        if (max < score) {
                            max = score;
                            index = c;
                        }
                    }
                    if (max < option.conf_thresh) continue;
				
                    float coordinate[4] ={0.f};
                    for (int n = 0; n < 4; ++n) {
                        int n_offset = n*16;
                        std::vector<float> softmaxed = softmax(buffer+h_offset+w_offset+n_offset, 16);
                        float sum = 0.f;
                        for (int dfln = 0; dfln < 16; ++dfln) {
                            sum += dfln*softmaxed[dfln];
                        }
                        coordinate[n] = sum;
                    } 
                    float x0 = static_cast<float>(w) - coordinate[0] + option.grid_offset;
                    float y0 = static_cast<float>(h) - coordinate[1] + option.grid_offset;
                    float x1 = static_cast<float>(w) + coordinate[2] + option.grid_offset;
                    float y1 = static_cast<float>(h) + coordinate[3] + option.grid_offset;
                    MResult result;
                    result.batch_idx  = n;
                    result.cls_idx    = index;
                    result.class_name = option.labels[index];
                    result.score      = max;
                    result.bbox.tl.x  = (x0*(option.strides[i])-context.pad_w)/context.scale;
					result.bbox.tl.y  = (y0*(option.strides[i])-context.pad_h)/context.scale;
					result.bbox.br.x  = (x1*(option.strides[i])-context.pad_w)/context.scale;
					result.bbox.br.y  = (y1*(option.strides[i])-context.pad_h)/context.scale;
                    __results.push_back(result);
                }
            }
            buffer += (option.grids[i]*option.grids[i]*(64+nc));
        }
        nms(__results, option.iou_thresh);
        results.insert(results.end(), __results.begin(), __results.end());
    }
    return results;
}

} // namespace mariana

