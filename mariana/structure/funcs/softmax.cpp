/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/softmax.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:16:07:45
 * Description:
 * 
 */

#include <structure/funcs/softmax.h>
#include <core/utils/logging.h>

namespace mariana {

tensor_list SoftmaxFunction::compute(tensor_list&& inputs) {
    
}

ShapeList SoftmaxFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now softmax only support 1 input:"<<shapes.size();
    const Shape& ishape = shapes[0];
    return {ishape};
}

} // namespace mariana
