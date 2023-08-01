/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/engine.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-01:08:43:42
 * Description:
 *
 */

#ifndef __ENGINE_EXECUTOR_H__
#define __ENGINE_EXECUTOR_H__

#include <core/utils/status.h>
#include <marc/marc.h>

namespace mariana {

class Engine {
public:
    Engine() {}
    virtual ~Engine() {}
    virtual Status build_external(Graph& graph, const ConvertContext& context) {
        return absl::UnimplementedError("Engine build_external method is not implemented");
    }
    virtual Status build_internal(Graph& graph, const ConvertContext& context) {
        return absl::UnimplementedError("Engine build_internal method is not implemented");
    }
    virtual Status de_serialize(Graph& graph, const ConvertContext& context)=0;
};
} // namespace mariana

#endif /* __ENGINE_EXECUTOR_H__ */

