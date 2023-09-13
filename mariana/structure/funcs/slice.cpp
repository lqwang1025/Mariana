/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : slice.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-03:11:27:25
 * Description:
 * 
 */

#include <structure/funcs/slice.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list SliceFunction::compute(tensor_list&& inputs) {
    
}

ShapeList SliceFunction::infer_shape(ShapeList shapes) {
    std::cout<<"dd:"<<option.begin<<" "<<option.end<<" "
             <<option.step<<" "<<option.axis<<std::endl;
}

} // namespace mariana

