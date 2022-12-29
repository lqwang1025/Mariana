/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/pool.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:21:33
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_POOL_H__
#define __STRUCTURE_FUNCS_POOL_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

enum class PoolType : uint8_t {
    None = 0,
        Max = 1,
        Ave = 2,
        GAve = 3
        
};

struct PoolOption : public BaseOption {
    PoolOption() {}
    ~PoolOption() {}
    PoolType type = PoolType::None;
};

struct PoolFunction : public Function {
    PoolFunction() {}
    ~PoolFunction() {}
    PoolOption option;
    tensor_list compute(tensor_list&& inputs) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_POOL_H__ */

