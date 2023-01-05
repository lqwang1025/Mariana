/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/function.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-23:09:35:57
 * Description:
 * 
 */

#include <structure/function.h>
#include <core/utils/logging.h>

namespace mariana {

float Function::compute_FLOPs(ShapeList oshapes) {
    MCHECK(oshapes.size() == 1);
    return static_cast<float>(oshapes[0].size() / 1024.f / 1024.f);
}

} // namespace mariana
