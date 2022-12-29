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

#include <core/impl/type.h>
#include <structure/ir.h>
#include <core/utils/logging.h>
#include <core/utils/status.h>
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
        GraphInfo() {}
        GraphInfo(const GraphInfo& rhs) {}
        GraphInfo& operator=(const GraphInfo& rhs) {
            if (this == &rhs) return *this;
            name = rhs.name;
            doc_string = rhs.doc_string;
            node_name_map = rhs.node_name_map;
            tensor_name_map = rhs.tensor_name_map;
            graph = rhs.graph;
            return *this;
        }
        std::string name = "";
        std::string doc_string = "";
        std::unordered_map<std::string, ::onnx::NodeProto*> node_name_map;
        std::unordered_map<std::string, ::onnx::TensorProto*> tensor_name_map;
        ::onnx::GraphProto* graph = nullptr;
    };
    struct NodeInfo {
        NodeInfo() {
            is_input = false;
            nodes.clear();
            tensors.clear();
        }
        NodeInfo(const NodeInfo& rhs) {
            this->operator=(rhs);
        }
        NodeInfo& operator=(const NodeInfo& rhs) {
            if (this == &rhs) return *this;
            is_input = rhs.is_input;
            nodes = rhs.nodes;
            tensors = rhs.tensors;
            return *this;
        }
        bool is_input = false;
        std::vector<::onnx::NodeProto*> nodes;
        std::vector<::onnx::TensorProto*> tensors;
    };
    explicit OnnxScope(const std::string& name);
    ::onnx::ModelProto onnx_model;
    ModelInfo model_info;
    GraphInfo graph_info;
    std::unordered_map<std::string, NodeInfo> nodes_info;
    static bool parse(const std::string& name, ::onnx::ModelProto& onnx_model);
    static Status sort_by_execution_order(const ::onnx::GraphProto& input_graph,
                                          ::onnx::GraphProto* output_graph);
    static std::unordered_map<std::string, OnnxScope::NodeInfo> init_nodes_info(
        const ::onnx::GraphProto& graph);
    static OnnxScope::GraphInfo init_graph_info(const ::onnx::GraphProto& graph);
private:
    void _init();
    
};

class OnnxConverter {
public:
    OnnxConverter()=default;
    virtual ~OnnxConverter()=default;
    virtual void run(const ::onnx::NodeProto&, Node&, const OnnxScope&) = 0;
};

class OnnxHolder final {
public:
    typedef std::unordered_map<OpCategory, OnnxConverter*> OnnxConverterMap;
    static OnnxConverterMap& get_onnx_converter_map() {
        static OnnxConverterMap* onnx_converter_map = new OnnxConverterMap;
        return *onnx_converter_map;
    }
    
    static void add_onnx_convert(const OpCategory& category, OnnxConverter* converter) {
        OnnxConverterMap& onnx_converter_map = get_onnx_converter_map();
        if (onnx_converter_map.count(category) == 1) {
            MVLOG(WARNING)<<"OP "<<category<<" had been registred.";
            return;
        }
        onnx_converter_map[category] = converter;
    }

    static OnnxConverter* search(const OpCategory& category) {
        OnnxConverterMap& onnx_converter_map = get_onnx_converter_map();
        if (onnx_converter_map.size() == 0) {
            MVLOG(FATAL)<<"There is no op in registry.";
            return nullptr;
        }
        if (onnx_converter_map.count(category) == 0) {
            return onnx_converter_map["Default"];
        }
        return onnx_converter_map[category];
    }
    
    static void release() {
        OnnxConverterMap& onnx_converter_map = get_onnx_converter_map();
        auto iter = onnx_converter_map.begin();
        while (iter != onnx_converter_map.end()) {
            if (iter->second != nullptr) {
                delete iter->second;
            }
            onnx_converter_map.erase(iter++);
        }
    }
private:
    OnnxHolder()=delete;
    OnnxHolder(const OnnxHolder&)=delete;
    OnnxHolder& operator=(const OnnxHolder&)=delete;

};

Graph* parse(const std::string& name);

#define DECLARE_ONNX_CONVERTER_CLASS(classname)                         \
    class classname final : public OnnxConverter {                      \
    public:                                                             \
        classname() {}                                                  \
        virtual ~classname()=default;                                   \
        virtual void run(const ::onnx::NodeProto&, Node&, const OnnxScope&) override; \
    }


}} // namespace mariana::onnx

#endif /* __MARC_ONNX_ONNX_H__ */

