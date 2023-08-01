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
#include <structure/funcs/tensorRT/trt_executor.h>

namespace mariana {

Graph* parse(const ConvertContext& context) {
    if (absl::EndsWith(context.model_path, ".onnx")) {
        if (context.back_end == Backend::TRT) {
            if (context.from_scratch) { // To construct network form onnx by us.
                Graph* graph = onnx::parse(context.model_path);
            } else { // To construct network form onnx by TRT.
                std::shared_ptr<trt::TensorRTEngine> engine{new trt::TensorRTEngine()};
                Graph* graph = new Graph{engine};
                MCHECK(engine->build_external(*graph, context).ok());
                return graph;
            }
        } else {
            MLOG(FATAL)<<"Unspport model type:"<<context.model_path
                       <<" Now support model from onnx:{TRT}";
        }
    } else if (absl::EndsWith(context.model_path, ".plan")) { // TRT
        std::shared_ptr<trt::TensorRTEngine> engine{new trt::TensorRTEngine()};
        Graph* graph = new Graph{engine};
        MCHECK(engine->de_serialize(*graph, context).ok());
        return graph;
    } else if (absl::EndsWith(context.model_path, ".rknn")) { // RKNN
        MLOG(FATAL)<<"TODO....";
    } else {
        MLOG(FATAL)<<"Unspport model type:"<<context.model_path
                   <<" Now support model:{*.plan(for TRT) *.rknn(for RKNN) *.onnx(for Internal)}";
    }
}

} // namespace mariana
