/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : math.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:28:42
 * Description:
 * 
 */

#include <core/utils/logging.h>
#include <structure/funcs/math.h>

namespace mariana {

tensor_list MathFunction::compute(tensor_list&& inputs) {
    
}

ShapeList MathFunction::infer_shape(ShapeList shapes) {
    if (shapes.size() == 2) {
        const Shape& ashape = shapes[0];
        const Shape& bshape = shapes[1];
        if (ashape.size() > bshape.size()) {
            return {ashape};
        } else {
            return {bshape};
        }
    } else if (shapes.size() == 1) {
        const Shape& ashape = shapes[0];
        return {ashape};
    } else {
        MLOG(FATAL)<<"Unsupport in Add shape inference input size:"<<shapes.size();
    }
    
}

} // namespace mariana
