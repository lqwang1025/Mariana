/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/gemm.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:33:46
 * Description:
 * 
 */

#include <structure/funcs/gemm.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana {

tensor_list GemmFunction::compute(tensor_list&& inputs) {
    
}

ShapeList GemmFunction::infer_shape(ShapeList shapes) {
    MCHECK(shapes.size() == 1)<<"Now Gemm only support 1 input:"<<shapes.size();
    const Shape& ishape = shapes[0];
    const Tensor& weight = option.weights[0];
    const Shape& kshape = weight.shape();
    if (option.trans_a && option.trans_b) {
        return {{ishape[1], kshape[0]}};
    } else if (option.trans_a && option.trans_b == false) {
        return {{ishape[1], kshape[1]}};
    } else if (false == option.trans_a && option.trans_b) {
        return {{ishape[0], kshape[0]}};
    } else { // false == option.trans_a && option.trans_b == false
        return {{ishape[0], kshape[1]}};
    }
}

float GemmFunction::compute_FLOPs(ShapeList oshapes) {
    // FLOPs = [I+(I-1)+1] x O = (2 x I) x O
    MCHECK(oshapes.size() == 1);
    const Tensor& weight = option.weights[0];
    const Shape& kshape = weight.shape();
    float flops = static_cast<float>(kshape[0]*2*kshape[1]);
    return flops/1024.f/1024.f;
}

} // namespace mariana
