/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops/add.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:51:11
 * Description:
 * 
 */

#include <marc/onnx/register.h>

namespace mariana { namespace onnx {

void AddConverter::run(const ::onnx::NodeProto&, const OnnxScope&) {
    std::cout<<__func__<<std::endl;
}

}} // namespace mariana::onnx
