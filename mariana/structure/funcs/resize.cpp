/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : resize.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:15:50:43
 * Description:
 * 
 */

#include <numeric>
#include <vector>

#include <structure/funcs/resize.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list ResizeFunction::compute(tensor_list&& inputs) {
    
}

ShapeList ResizeFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now reshape only support 1 input:"<<shapes.size();
}

} // namespace mariana
