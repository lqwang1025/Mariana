/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/resize.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-21:15:44:55
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_RESIZE_H__
#define __STRUCTURE_FUNCS_RESIZE_H__

#include <vector>
#include <cstdint>

#include <structure/tensor.h>
#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

enum class ResizeCoordinateTransformationMode : uint8_t {
    None = 0,
    AlignCorners = 1,
    Asymmetric = 2,
    HalfPixel = 3
};

enum class ResizeMode : uint8_t {
    None = 0,
    Nearest = 1,
    Linear = 2
};

enum class ResizeRoundMode : uint8_t {
    None = 0,
    //! Round half up.
    HalfUp = 1,
    //! Round half down.
    HalfDown = 2,
    //! Round to floor.
    Floor = 3,
    //! Round to ceil.
    Ceil = 4,
};

struct ResizeOption : public BaseOption {
    ResizeOption() {}
    ~ResizeOption() {}
    ResizeMode resize_mode = ResizeMode::None;
    ResizeCoordinateTransformationMode resize_coordinate_transformation_mode = ResizeCoordinateTransformationMode::None;
    ResizeRoundMode resize_round_mode = ResizeRoundMode::None;
    std::vector<float> scales;
};

struct ResizeFunction : public Function {
    ResizeFunction() {}
    ~ResizeFunction() {}
    ResizeOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
    float compute_FLOPs(ShapeList oshapes) override {
        return 0.f;
    }
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_RESIZE_H__ */

