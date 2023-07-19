/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : default.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:17:15:34
 * Description:
 * 
 */

#include <marc/onnx/register.h>

namespace mariana { namespace onnx {

void DefaultConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
}

}} // namespace mariana::onnx

