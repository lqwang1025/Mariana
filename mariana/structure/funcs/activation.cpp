/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : activation.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:09:47
 * Description:
 * 
 */

#include <structure/funcs/activation.h>
#include <core/utils/logging.h>

namespace mariana {

tensor_list ActivationFunction::compute(tensor_list&& inputs) {
    
}

ShapeList ActivationFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now Act only support 1 input:"<<shapes.size();
    const Shape& ishape = shapes[0];
    return {ishape};
}

} // namespace mariana
