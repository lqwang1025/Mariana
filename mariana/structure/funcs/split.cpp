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
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list SplitFunction::compute(tensor_list&& inputs) {
    
}

ShapeList SplitFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now split only support 1 input:"<<shapes.size();
    int32_t axis = option.axis;
    std::vector<int32_t> split = option.split;
    Shape ishape = shapes[0];
    if (split.empty() == false) {
        int input_slice_num = ishape[axis];
        int sum_check = 0;
        for (size_t i = 0; i < split.size(); ++i) {
            sum_check += split[i];
        }
        MCHECK(sum_check==input_slice_num)<<"Mar Fatal: Infer shape for Split failed"
                                          <<sum_check<<"vs."<<input_slice_num;
        ShapeList oshapes;
        for (size_t i = 0; i < split.size(); ++i) {
            Shape _ishape = ishape;
            _ishape[axis] = split[i];
            oshapes.push_back(_ishape);
        }
        return oshapes;
    } else {
        MLOG(FATAL)<<"Unsupport in split empty";
    }
}

} // namespace mariana

