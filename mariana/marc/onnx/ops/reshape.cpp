/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : reshape.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:00
 * Description:
 * 
 */

#include <numeric>

#include <core/utils/arrary_ref.h>
#include <marc/onnx/register.h>
#include <structure/funcs/reshape.h>
#include <marc/onnx/proto/onnx_help.h>

namespace mariana { namespace onnx {

void ReshapeConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    ReshapeFunction* func = static_cast<ReshapeFunction*>(dst.op());
    const ::onnx::TensorProto* shape = scope.nodes_info.at(src.name()).tensors[0];
    std::vector<int64_t> _shape;
    void* content;
    get_content_from_tensor(*shape, _shape, &content);
    int64_t product = std::accumulate(_shape.begin(), _shape.end(),
                                      1, std::multiplies<int64_t>());
    func->option.shape.resize(product);
    memcpy(func->option.shape.data(), content, sizeof(int64_t)*product);
}

}} // namespace mariana::onnx
