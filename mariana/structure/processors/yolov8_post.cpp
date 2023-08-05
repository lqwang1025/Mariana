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
    result_list results;
    float* buffer = static_cast<float*>(inputs[0].data());
	float* cls_buf = static_cast<float*>(inputs[1].data());
    Shape shape1 = inputs[0].shape(); // [1, 8400, 4]
    Shape shape2 = inputs[1].shape(); // [1, 8400, 17]
    int nb = shape1[0];
    int ng = shape1[2];
    int nc = shape2[2];
    for (int n = 0; n < nb; ++n) {
        result_list __results;
        for (size_t i = 0; i < option.grids.size(); ++i) {
            for (int h = 0; h < option.grids[i]; ++h) {
                int h_offset = (h*option.grids[i])<<2;
                int cls_h_offset = h*option.grids[i]*nc;
                for (int w = 0; w < option.grids[i]; ++w) {
                    int w_offset = w<<2;
                    int cls_w_offset = w*nc;
                    float max = -FLT_MAX;
                    int index = -1;
                    for (int c = 0; c < nc; ++c) {
                        float score = cls_buf[cls_h_offset+cls_w_offset+c];
                        if (max < score) {
                            max = score;
                            index = c;
                        }
                    }
                    if (max < option.conf_thresh) continue;
				
                    float x0 = buffer[h_offset+w_offset+0];
                    float y0 = buffer[h_offset+w_offset+1];
                    float x1 = buffer[h_offset+w_offset+2];
                    float y1 = buffer[h_offset+w_offset+3];
                    x0 = static_cast<float>(w) - x0 + option.grid_offset;
                    y0 = static_cast<float>(h) - y0 + option.grid_offset;
                    x1 = static_cast<float>(w) + x1 + option.grid_offset;
                    y1 = static_cast<float>(h) + y1 + option.grid_offset;
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
            buffer += (option.grids[i]*option.grids[i]<<2);
            cls_buf += (option.grids[i]*option.grids[i]*nc);
        }
        nms(__results, option.iou_thresh);
        results.insert(results.end(), __results.begin(), __results.end());
    }
    return results;
}

} // namespace mariana

