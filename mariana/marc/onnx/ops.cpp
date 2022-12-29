/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:04:55
 * Description:
 * 
 */

#include <marc/onnx/ops.h>

namespace mariana { namespace onnx {

std::set<std::string> CONTINUE_OP = {
    KIdentity
};

}} // namespace mariana::onnx
