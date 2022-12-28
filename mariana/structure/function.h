/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/function.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:12:09
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCTION_H__
#define __STRUCTURE_FUNCTION_H__

#include <vector>
#include <mutex>
#include <structure/edge.h>
#include <structure/tensor.h>

namespace mariana {

using tensor_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

struct Function {
    explicit Function(uint64_t sequence_nr, edge_list&& next_edges = edge_list()) : sequence_nr_(sequence_nr), next_edges_(std::move(next_edges)) {}
    Function(const Function& other) = delete;
    Function(Function&& other) = delete;
    Function& operator=(const Function& other) = delete;
    Function& operator=(Function&& other) = delete;
    virtual ~Function() = default;
protected:
    uint64_t thread_id_ = 0;
    mutable bool has_parent_ = false;
    uint64_t topological_nr_ = 0;
    const uint64_t sequence_nr_;
    std::mutex mutex_;
    edge_list next_edges_;
    
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCTION_H__ */

