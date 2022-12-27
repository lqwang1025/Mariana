/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/marc.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-26:17:40:34
 * Description:
 * 
 */

#include <marc/marc.h>
#include <marc/onnx/onnx.h>

namespace mariana {

bool parse(const std::string& name) {
    return onnx::parse(name);
}

} // namespace mariana
