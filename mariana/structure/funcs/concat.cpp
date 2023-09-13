/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : concat.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-20:17:33:01
 * Description:
 * 
 */

#include <structure/funcs/concat.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list ConcatFunction::compute(tensor_list&& inputs) {
        
}

ShapeList ConcatFunction::infer_shape(ShapeList shapes) {
    int32_t axis = option.axis;
    Shape ishape = shapes[0];
    int32_t new_dims = 0;
    for(auto& it : shapes) {
        new_dims += it[axis];
    }
    ishape[axis] = new_dims;
    return {ishape};
}

} // namespace mariana

