/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : maro/transform.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:16:33:36
 * Description:
 *
 */

#ifndef __MARO_TRANSFORM_H__
#define __MARO_TRANSFORM_H__

#include <string>
#include <vector>
#include <structure/ir.h>

namespace mariana { namespace transform {

typedef std::vector<std::string> TransformParameters;

void transform(Graph& graph, const TransformParameters& params);

}} // namespace mariana::transform

#endif /* __MARO_TRANSFORM_H__ */

