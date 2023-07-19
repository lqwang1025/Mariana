/*
 *        (C) COPYRIGHT Daniel Limited.
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
        Avg = 2,
        GAvg = 3
        
};

struct PoolOption : public BaseOption {
    PoolOption() {
        kernel_shape.clear();
        pads.clear();
        strides.clear();
    }
    ~PoolOption() {
        kernel_shape.clear();
        pads.clear();
        strides.clear();
    }
    PoolType type = PoolType::None;
    int32_t ceil_mode = 0;
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> pads;
    std::vector<int32_t> strides;
};

struct PoolFunction : public Function {
    PoolFunction() {}
    ~PoolFunction() {}
    PoolOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_POOL_H__ */

