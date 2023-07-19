/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/optimize/transform.h
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-03:11:05:38
 * Description:
 *
 */

#ifndef __MARC_ONNX_OPTIMIZE_TRANSFORM_H__
#define __MARC_ONNX_OPTIMIZE_TRANSFORM_H__

#include <vector>
#include <marc/onnx/onnx.h>

namespace mariana { namespace onnx { namespace transform {

typedef std::vector<std::string> TransformParameters;

void transform(OnnxScope& scope, const TransformParameters& params);

}}} // mariana::onnx::transform

#endif /* __MARC_ONNX_OPTIMIZE_TRANSFORM_H__ */

