/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/reduce.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-25:09:04:22
 * Description:
 * 
 */

#include <structure/funcs/reduce.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list ReduceFunction::compute(tensor_list&& inputs) {
    
}

ShapeList ReduceFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now reduce only support 1 input.";
    ArrayRef<int32_t> axes = option.axes;
    bool keepdims = option.keepdims;
    const Shape& ishape = shapes[0];
    int n = ishape[0];
    std::vector<int32_t> oshape;
    for (auto& it : ishape.data()) {
        oshape.push_back(it);
    }
    for (auto& it : axes) {
        if (keepdims == true) {
            oshape[it] = 1;
        } else {
            oshape[it] = -1;
        }
    }
    for (auto it = oshape.begin(); it != oshape.end();) {
        if (*it == -1)
            it = oshape.erase(it);
        else {
            ++it;
        }
    }
    
    Shape rshape(oshape);
    return {rshape};
}

} // namespace mariana

