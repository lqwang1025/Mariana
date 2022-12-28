/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : relu.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:38
 * Description:
 * 
 */

#include <marc/onnx/register.h>

namespace mariana { namespace onnx {

void ReluConverter::run(const ::onnx::NodeProto&, Node&, const OnnxScope&) {
    
}

}} // namespace mariana::onnx

