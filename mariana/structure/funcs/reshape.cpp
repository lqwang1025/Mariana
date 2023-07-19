/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : reshape.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:10:36:15
 * Description:
 * 
 */

#include <numeric>
#include <vector>

#include <structure/funcs/reshape.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list ReshapeFunction::compute(tensor_list&& inputs) {
    
}

ShapeList ReshapeFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now reshape only support 1 input:"<<shapes.size();
    const Shape& ishape = shapes[0];
    ArrayRef<int64_t> shape = option.shape;
    int64_t product = std::accumulate(option.shape.begin(), option.shape.end(),
                                      1, std::multiplies<int64_t>());
    std::vector<int64_t> oshape;
    oshape.resize(shape.size());
    if (product < 0) {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != -1) {
                oshape[i] = shape[i];
            } else {
                int64_t oproduct = std::accumulate(oshape.begin(), oshape.begin()+i,
                                                   1, std::multiplies<int64_t>());
                oshape[i] = ishape.size()/oproduct;
            }
        }
        return {ArrayRef<int64_t>{oshape}};
    } else if (product == ishape.size()) {
        return {shape};
    } else {
        MCHECK(false)<<"Reshape size is not euqal:"<<product<<" "<<ishape.size();
    }
}

} // namespace mariana
