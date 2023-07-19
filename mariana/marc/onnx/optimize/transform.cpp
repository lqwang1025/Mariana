/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/optimize/transform.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-03:11:06:51
 * Description:
 * 
 */

#include <core/utils/logging.h>
#include <marc/onnx/optimize/transform_utils.h>
#include <marc/onnx/optimize/transform.h>

namespace mariana { namespace onnx { namespace transform {

void transform(OnnxScope& scope, const TransformParameters& params) {
    for (auto& param : params) {
        TransformRegistry* reg = transform::get_transform_registry();
        if (reg->count(param) == 0) {
            MCHECK(false)<<"There no optimized function:"<<param<<".";
        }
        (*reg)[param](scope);
    }
}

}}} // mariana::onnx::transform
