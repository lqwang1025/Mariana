/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/onnx.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-26:17:42:35
 * Description:
 *
 */

#ifndef __MARC_ONNX_ONNX_H__
#define __MARC_ONNX_ONNX_H__

#include <string>
#include <cstdint>
#include <unordered_map>
#include <marc/onnx/proto/onnx.pb.h>

namespace mariana { namespace onnx {

struct OnnxScope {
    struct ModelInfo {
        int64_t ir_version = 0;
        std::string producer_name = "";
        std::string producer_version = "";
        std::string domain = "";
        int64_t model_version = 0;
        std::string doc_string = "";
    };
    struct GraphInfo {
        std::string name = "";
        std::string doc_string = "";
        std::unordered_map<std::string, ::onnx::NodeProto*> node_name_map;
        std::unordered_map<std::string, ::onnx::TensorProto*> tensor_name_map;
        ::onnx::GraphProto* graph;
        
    };
    explicit OnnxScope(const std::string& name);
    ::onnx::ModelProto onnx_model;
    ModelInfo model_info;
    GraphInfo graph_info;
    static bool parse(const std::string& name, ::onnx::ModelProto& onnx_model);
private:
    void _init();
};

class OnnxConverter {
public:
    OnnxConverter()=default;
    virtual ~OnnxConverter()=default;
    virtual void run(const ::onnx::NodeProto&, const OnnxScope&) = 0;
};
using OpCategory = std::string;


bool parse(const std::string& name);

}} // namespace mariana::onnx

#endif /* __MARC_ONNX_ONNX_H__ */

