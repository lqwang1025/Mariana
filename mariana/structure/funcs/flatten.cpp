/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : flatten.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:31:52
 * Description:
 * 
 */

#include <core/utils/arrary_ref.h>
#include <core/utils/logging.h>
#include <structure/funcs/flatten.h>

namespace mariana {

tensor_list FlattenFunction::compute(tensor_list&& inputs) {
    
}

ShapeList FlattenFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now flatten only support 1 input.";
    int32_t axis = option.axis;
    const Shape& ishape = shapes[0];
    int32_t product = 1;
    for (int i = axis; i < ishape.dims(); ++i) {
        product *= ishape[i];
    }
    std::vector<int32_t> oshape;
    oshape.reserve(ishape.dims());
    for (int i = 0; i < axis; ++i) {
        oshape.push_back(ishape[i]);
    }
    oshape.push_back(product);
    Shape rshape(oshape);
    return {rshape};
}

} // namespace mariana {
