/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/proto/onnx_help.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:46:05
 * Description:
 *
 */

#ifndef __MARC_ONNX_PROTO_ONNX_HELP_H__
#define __MARC_ONNX_PROTO_ONNX_HELP_H__

#include <string>
#include <vector>

#include <core/utils/logging.h>

namespace onnx {
class NodeProto;
class TensorProto;
} // namespace onnx

namespace mariana { namespace onnx {

bool node_has_attr(const ::onnx::NodeProto& node, const std::string&& name);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::string *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, float *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, int64_t *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, int32_t *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<std::string> *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<float> *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<int64_t> *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<int32_t> *value);
bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, ::onnx::TensorProto *value);

void get_content_from_tensor(const ::onnx::TensorProto& tensor, std::vector<int64_t>& shape, void** content);

#define GET_ONNX_NODE_ATTR(__node, _name, value_ptr)                    \
    do {                                                                \
        bool ret = get_node_attr(__node, std::move(_name), value_ptr); \
        MCHECK(ret)<<"Get node["<<__node.name()<<"] value failed.";    \
    } while(false)

}} // namespace mariana::onnx

#endif /* __MARC_ONNX_PROTO_ONNX_HELP_H__ */

