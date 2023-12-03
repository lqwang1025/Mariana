/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/proto/onnx_help.cc
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:46:05
 * Description:
 *
 */

#include <marc/onnx/proto/onnx_help.h>
#include <marc/onnx/proto/onnx.pb.h>

namespace mariana { namespace onnx {

#define GET_NODE_ATTR(type, exp)                                        \
    for (auto attr : node.attribute()) {                                \
        if (attr.name() == name) {                                      \
            if (attr.type() == type) {                                  \
                exp(attr);                                              \
                return true;                                            \
            }                                                           \
        }                                                               \
    }                                                                   \
    return false                                                        \
    

bool node_has_attr(const ::onnx::NodeProto& node, const std::string&& name) {
    for (auto attr : node.attribute()) {
        if (attr.name() == name) {
            return true;
        }
    }
    return false;
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::string *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {*value = attr.s();};
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, float *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {*value = attr.f();};
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, int64_t *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {*value = attr.i();};
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, int32_t *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {
        //TODO: waringing  here
        *value = static_cast<int32_t>(attr.i());
    };
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<std::string> *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {
        value->reserve(attr.strings_size());
        for (auto it : attr.strings()) {
            value->push_back(it);
        }
    };
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<float> *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {
        value->reserve(attr.floats_size());
        for (auto it : attr.floats()) {
            value->push_back(it);
        }
    };
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<int64_t> *value){
    auto func = [&](const ::onnx::AttributeProto& attr) {
        value->reserve(attr.ints_size());
        for (auto it : attr.ints()) {
            value->push_back(it);
        }
    };
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, std::vector<int32_t> *value){
    auto func = [&](const ::onnx::AttributeProto& attr) {
        value->reserve(attr.ints_size());
        for (auto it : attr.ints()) {
            value->push_back(static_cast<int32_t>(it));
        }
    };
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS;
    GET_NODE_ATTR(type, func);
}

bool get_node_attr(const ::onnx::NodeProto& node, const std::string&& name, ::onnx::TensorProto *value) {
    auto func = [&](const ::onnx::AttributeProto& attr) {
        *value = attr.t();
    };
    auto type = ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR;
    GET_NODE_ATTR(type, func);
}

void get_content_from_tensor(const ::onnx::TensorProto& tensor, std::vector<int64_t>& shape, void** content) {
    shape.clear();
    shape.reserve(tensor.dims_size());
    for (auto dim : tensor.dims()) {
        shape.push_back(dim);
    }
    if (tensor.dims_size() == 0) {
        shape.push_back(1);
    }
    *content = static_cast<void*>(const_cast<char*>(tensor.raw_data().data()));
}

#undef GET_NODE_ATTR

}} // namespace mariana::onnx



