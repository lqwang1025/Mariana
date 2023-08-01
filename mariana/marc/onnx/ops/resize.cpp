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

#include <marc/onnx/register.h>
#include <structure/funcs/resize.h>

namespace mariana { namespace onnx {

void ResizeConverter::run(const ::onnx::NodeProto&, Node&, const OnnxScope&) {
    std::cout<<"sssssssssss"<<std::endl;
    // ReshapeFunction* func = static_cast<ReshapeFunction*>(dst.op());
    // const ::onnx::TensorProto* shape = scope.nodes_info.at(src.name()).tensors[0];
    // std::vector<int64_t> _shape;
    // void* content;
    // get_content_from_tensor(*shape, _shape, &content);
    // int64_t product = std::accumulate(_shape.begin(), _shape.end(),
    //                                   1, std::multiplies<int64_t>());
    // func->option.shape.resize(product);
    // memcpy(func->option.shape.data(), content, sizeof(int64_t)*product);
}

}} // namespace mariana::onnx
