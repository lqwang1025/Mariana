/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : permute.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:16:11:08
 * Description:
 * 
 */

#include <structure/funcs/permute.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>
#include <core/macros/macros.h>

namespace mariana {

tensor_list PermuteFunction::compute(tensor_list&& inputs) {
    
}

ShapeList PermuteFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now permute only support 1 input:"<<shapes.size();
    std::vector<int32_t> perm = option.perm;
    const Shape& ishape = shapes[0];

    Shape oshape = ishape;
    for (size_t i = 0; i < perm.size(); ++i) {
        oshape[i] = ishape[perm[i]];
    }
    return {oshape};
    
}

} // namespace mariana
