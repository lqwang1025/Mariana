/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/pool.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:22:10
 * Description:
 * 
 */

#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>
#include <structure/funcs/pool.h>
#include <core/macros/macros.h>

namespace mariana {

tensor_list PoolFunction::compute(tensor_list&& inputs) {
    
}

ShapeList PoolFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now pooling only support 1 input:"<<shapes.size();
    ArrayRef<int32_t> kernel_shape = option.kernel_shape;
    ArrayRef<int32_t> pads = option.pads;
    ArrayRef<int32_t> strides = option.strides;
    int32_t ceil_mode = option.ceil_mode;
    PoolType type = option.type;
    const Shape& shape = shapes[0];
    int batch = shape[0];
    int channel = shape[1];
    int input_h = shape[2];
    int input_w = shape[3];
    int output_h, output_w;
    if (kernel_shape == ArrayRef<int32_t>{input_h, input_w} &&
        pads == ArrayRef<int32_t>{0, 0, 0, 0}) {
        option.type = PoolType::GAvg;
    }

    if (option.type == PoolType::GAvg) {
        option.pads = {0, 0, 0, 0};
        option.kernel_shape = {input_h, input_w};
        option.strides = {1, 1};
        output_h = 1;
        output_w = 1;
    } else { // pads : t b l r
        input_w += (pads[2] + pads[3]);
        input_h += (pads[0] + pads[1]);
        if (ceil_mode == 1) {
            output_w = UP_DIV(input_w - kernel_shape[1], strides[1]) + 1;
            output_h = UP_DIV(input_h - kernel_shape[0], strides[0]) + 1;
        } else {
            output_w = floor((input_w - kernel_shape[1]) / strides[1] + 1);
            output_h = floor((input_h - kernel_shape[0]) / strides[0] + 1);
        }
    }
    return {{batch, channel, output_h, output_w}};
}

} // namespace mariana
