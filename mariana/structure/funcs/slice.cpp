/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : slice.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-03:11:27:25
 * Description:
 * 
 */

#include <structure/funcs/slice.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list SliceFunction::compute(tensor_list&& inputs) {
    
}

ShapeList SliceFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now Slice only support 1 input:"<<shapes.size();
    Shape ishape = shapes[0];
    int32_t axis = option.axis;
    // Check: axis must be in the range: [-input->dim_num, input->dim_num)
    MCHECK(axis < -ishape.dims() || ishape.dims() <= axis)<<"Input slcie axis not to be supported.";
    option.axis += ishape.dims();
    option.axis %= ishape.dims();
    axis = option.axis;

    std::vector<int32_t> oshape;
    oshape.resize(ishape.dims());
    for (size_t i = 0; i < ishape.dims(); ++i) {
        if (axis == i) {
            int slice_end = option.end;
            if (option.end > ishape[i]) {
                slice_end = ishape[i];
                option.end = slice_end;
            }
            if (slice_end > 0) {
                oshape[i] = slice_end - option.begin;
                if (option.step > 1) {
                    oshape[i] = (oshape[i] - 1) / option.step + 1;
                }
            } else {
                oshape[i] = ishape[i] + slice_end - option.begin;
                if (option.step > 1) {
                    oshape[i] = (oshape[i] - 1) / option.step + 1;
                }
            }
            if (0 == oshape[i]) {
                oshape[i] = ishape[i];
            }
        } else {
            oshape[i] = ishape[i];
        }
    }
    Shape rshape(oshape);
    return {rshape};
}

} // namespace mariana

