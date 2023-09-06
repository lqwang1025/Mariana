/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : resize.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:07
 * Description:
 * 
 */

#include <numeric>

#include <marc/onnx/register.h>
#include <structure/funcs/resize.h>
#include <marc/onnx/proto/onnx_help.h>
#include <core/utils/logging.h>

namespace mariana { namespace onnx {

void ResizeConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    ResizeFunction* func = static_cast<ResizeFunction*>(dst.op());
    std::string coordinate_transformation_mode = "";
    GET_ONNX_NODE_ATTR(src, "coordinate_transformation_mode", &coordinate_transformation_mode);

    if (coordinate_transformation_mode == "asymmetric") {
        func->option.resize_coordinate_transformation_mode = ResizeCoordinateTransformationMode::Asymmetric;
    } else if (coordinate_transformation_mode == "half_pixel") {
        func->option.resize_coordinate_transformation_mode = ResizeCoordinateTransformationMode::HalfPixel;
    } else if (coordinate_transformation_mode == "align_corners") {
        func->option.resize_coordinate_transformation_mode = ResizeCoordinateTransformationMode::AlignCorners;
    } else {
        MLOG(FATAL)<<"Unsupport resize trans mode:"<<coordinate_transformation_mode;
    }
    
    std::string mode = "";
    GET_ONNX_NODE_ATTR(src, "mode", &mode);

    if (mode == "nearest" || mode == "") {
        func->option.resize_mode = ResizeMode::Nearest;
    } else if (mode == "linear") {
        func->option.resize_mode = ResizeMode::Linear;
    } else {
        MLOG(FATAL)<<"Unsupport resize mode:"<<coordinate_transformation_mode;
    }

    std::string nearest_mode = "";
    GET_ONNX_NODE_ATTR(src, "nearest_mode", &nearest_mode);

    if (nearest_mode == "floor") {
        func->option.resize_round_mode = ResizeRoundMode::Floor;
    } else if (nearest_mode == "ceil") {
        func->option.resize_round_mode = ResizeRoundMode::Ceil;
    }

    const ::onnx::TensorProto* shape = scope.nodes_info.at(src.name()).tensors[1];
    std::vector<int64_t> _shape;
    void* content;
    get_content_from_tensor(*shape, _shape, &content);
    int64_t product = std::accumulate(_shape.begin(), _shape.end(),
                                      1, std::multiplies<int64_t>());
    func->option.scales.resize(product);
    memcpy(func->option.scales.data(), content, sizeof(float)*product);
}

}} // namespace mariana::onnx
