/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/edge.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:11:47
 * Description:
 *
 */

#ifndef __STRUCTURE_EDGE_H__
#define __STRUCTURE_EDGE_H__

#include <cstdint>
#include <memory>

namespace mariana {

struct Function;

struct Edge {
    Edge() : function(nullptr), input_nr(0) {}
    
    bool is_valid() const {
        return function != nullptr;
    }
    bool operator==(const Edge& other) const {
        return this->function == other.function && this->input_nr == other.input_nr;
    }
    bool operator!=(const Edge& other) const {
        return !(*this == other);
    }
    std::shared_ptr<Function> function;
    uint32_t input_nr;
};

} // namespace mariana

#endif /* __STRUCTURE_EDGE_H__ */

