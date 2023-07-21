/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : transform.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:16:33:40
 * Description:
 * 
 */

#include <maro/transform.h>
#include <maro/transform_utils.h>

namespace mariana { namespace transform {

void transform(Graph& graph, const TransformParameters& params) {
    for (auto& param : params) {
        TransformRegistry* reg = get_transform_registry();
        if (reg->count(param) == 0) {
            MCHECK(false)<<"There no optimized function:"<<param<<".";
        }
        (*reg)[param](graph);
    }
}

}} // namespace mariana::transform
