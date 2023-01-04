/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/conv.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:09:45:10
 * Description:
 * 
 */

#include <core/utils/arrary_ref.h>
#include <core/utils/logging.h>
#include <structure/funcs/conv.h>

namespace mariana {

tensor_list ConvFunction::compute(tensor_list&& inputs) {
    
}

ShapeList ConvFunction::infer_shape(ShapeList shapes) {
    ArrayRef<int32_t> dilations = option.dilations;
    int32_t group = 1;
    ArrayRef<int32_t> kernel_shape = option.kernel_shape;
    ArrayRef<int32_t> pads = option.pads;
    ArrayRef<int32_t> strides = option.strides;
}

} // namespace mariana
