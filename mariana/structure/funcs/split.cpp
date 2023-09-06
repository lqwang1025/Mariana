/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : split.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-20:17:07:31
 * Description:
 * 
 */

#include <structure/funcs/split.h>

namespace mariana {

tensor_list SplitFunction::compute(tensor_list&& inputs) {
    
}

ShapeList SplitFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now split only support 1 input:"<<shapes.size();
    const Shape& ishape = shapes[0];
    int32_t axis = option.axis;
    
    
    return {ishape};
}

} // namespace mariana

