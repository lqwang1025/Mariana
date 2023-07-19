/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : add.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:28:42
 * Description:
 * 
 */

#include <core/utils/logging.h>
#include <structure/funcs/add.h>

namespace mariana {

tensor_list AddFunction::compute(tensor_list&& inputs) {
    
}

ShapeList AddFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 2)<<"Now add only support 2 input:"<<shapes.size();
    const Shape& ashape = shapes[0];
    const Shape& bshape = shapes[1];
    MCHECK(ashape == bshape)<<"A shape must equal with b shape :"<<ashape<<" "<<bshape;
    return {ashape};
}

} // namespace mariana
