/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/marc.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-26:17:40:34
 * Description:
 * 
 */

#include <marc/marc.h>
#include <marc/onnx/onnx.h>
#include <absl/strings/match.h>
#include <core/utils/logging.h>
#include <structure/processors/register.h>
#include <structure/processor.h>
#include <structure/graph_exec.h>
#include <maro/transform.h>

#ifdef WITH_TRT
#include <structure/funcs/tensorRT/trt_executor.h>
#endif

#ifdef WITH_RKNN
#include <structure/funcs/rknn/rknn_executor.h>
#endif

namespace mariana {

static void _attach_graph_with_post_processor(const proto::ModelInfo& model_info, Graph* graph) {
    if (model_info.post_process_category() == proto::PostProcessorCategory::POST_NONE) return;
    register_processors();
    auto pro_make = ProcessorHolder::search(model_info.post_process_category());
    MCHECK(pro_make!=nullptr)<<"There is no pro in registry:"<<model_info.post_process_category();
    graph->set_pro(pro_make(model_info));
    unregister_processors();
}

Graph* parse(const proto::ModelInfo& model_info) {
    if (absl::EndsWith(model_info.model_path(), ".onnx")) {
        if (model_info.back_end() == proto::BackEndType::BACKEND_TRT) {
#ifdef WITH_TRT
            if (model_info.from_scratch()) { // To construct network form onnx by us.
                std::shared_ptr<trt::TensorRTEngine> engine{new trt::TensorRTEngine()};
                Graph* graph = onnx::parse(model_info.model_path());
                GraphExec ge;
                ge.pre_run(*graph, model_info);
                transform::transform(*graph,{"trt_split_to_slice","base_fold_reshape_to_node", "trt_softmax_io_reshape"});
                MCHECK(engine->build_internal(*graph, model_info).ok());
                graph->set_engine(engine);
                _attach_graph_with_post_processor(model_info, graph);
                return graph;
            } else { // To construct network form onnx by TRT.
                std::shared_ptr<trt::TensorRTEngine> engine{new trt::TensorRTEngine()};
                Graph* graph = new Graph{engine};
                MCHECK(engine->build_external(*graph, model_info).ok());
                _attach_graph_with_post_processor(model_info, graph);
                return graph;
            }
#else
            MLOG(FATAL)<<"Mariana compiling is not with TRT!";
#endif
        } else {
            MLOG(FATAL)<<"Unspport model type:"<<model_info.model_path()
                       <<" Now support model from onnx:{TRT}";
        }
    } else if (absl::EndsWith(model_info.model_path(), ".plan")) { // TRT
#ifdef WITH_TRT
        std::shared_ptr<trt::TensorRTEngine> engine{new trt::TensorRTEngine()};
        Graph* graph = new Graph{engine};
        MCHECK(engine->de_serialize(*graph, model_info).ok());
        _attach_graph_with_post_processor(model_info, graph);
        return graph;
#else
            MLOG(FATAL)<<"Mariana compiling is not with TRT!";
#endif
    } else if (absl::EndsWith(model_info.model_path(), ".rknn")) { // RKNN
#ifdef WITH_RKNN
        std::shared_ptr<rknn::RknnEngine> engine{new rknn::RknnEngine()};
        Graph* graph = new Graph{engine};
        MCHECK(engine->de_serialize(*graph, model_info).ok());
        _attach_graph_with_post_processor(model_info, graph);
        return graph;
#else
        MLOG(FATAL)<<"Mariana compiling is not with RKNN!";
#endif
    } else {
        MLOG(FATAL)<<"Unspport model type:"<<model_info.model_path()
                   <<" Now support model:{*.plan(for TRT) *.rknn(for RKNN) *.onnx(for Internal)}";
    }
}

} // namespace mariana
