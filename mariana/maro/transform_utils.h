/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : maro/transform_utils.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:15:11:36
 * Description:
 *
 */

#ifndef __MARO_TRANSFORM_UTILS_H__
#define __MARO_TRANSFORM_UTILS_H__

#include <string>
#include <vector>

namespace mariana { namespace transform {

struct OpTypePattern {
    std::string op;
    std::vector<OpTypePattern> inputs;
};

}} // namespace mariana::transform

#endif /* __MARO_TRANSFORM_UTILS_H__ */

