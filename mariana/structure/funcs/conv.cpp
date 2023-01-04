/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/conv.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:09:45:10
 * Description:
 * 
 */

#include <core/utils/arrary_ref.h>
#include <core/utils/logging.h>
#include <structure/funcs/conv.h>

namespace mariana {

tensor_list ConvFunction::compute(tensor_list&& inputs) {
    
}

ShapeList ConvFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now convolution only support 1 input.";
    ArrayRef<int32_t> dilations = option.dilations;
    int32_t group = 1;
    ArrayRef<int32_t> kernel_shape = option.kernel_shape;
    ArrayRef<int32_t> pads = option.pads;
    ArrayRef<int32_t> strides = option.strides;
    const Shape& ishape = shapes[0];
    int n = ishape[0];
    int h = ishape[2], w = ishape[3];
    int out_c = option.oc;
    int out_h, out_w;

    /* handle the same padding case, which pad_h0 and pad_h1 is -1 (SAME_UPPER)
       -2 (SAME_LOWER) */
    if (pads[0] < 0) {
        out_h = (h - 1) / kernel_shape[0] + 1;
        int total_len = (out_h - 1) * strides[0] + kernel_shape[0];

        int pad_num = total_len - h;

        if (pads[0] == -1) {
            option.pads[0] = pad_num / 2;
            option.pads[1] = pad_num - pad_num / 2;
        } else {
            option.pads[1] = pad_num / 2;
            option.pads[0] = pad_num - pad_num / 2;
        }
    } else {
        out_h = (h - dilations[0] * (kernel_shape[0] - 1) - 1 + pads[0] + pads[1]) / strides[0] + 1;
    }

    if (pads[2] < 0) {
        out_w = (w - 1) / kernel_shape[1] + 1;
        int total_len = (out_w - 1) * strides[1] + kernel_shape[1];

        int pad_num = total_len - w;

        if (pads[2] == -1) {
            option.pads[2] = pad_num / 2;
            option.pads[3] = pad_num - pad_num / 2;
        } else {
            option.pads[3] = pad_num / 2;
            option.pads[2] = pad_num - pad_num / 2;
        }
    } else {
        out_w = (w - dilations[1] * (kernel_shape[1] - 1) - 1 + pads[2] + pads[3]) / strides[1] + 1;
    }
    return {{n, out_c, out_h, out_w}};
}

} // namespace mariana
