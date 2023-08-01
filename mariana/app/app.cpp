/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : app.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:08:57:10
 * Description:
 * 
 */

#include <iostream>
#include <marc/marc.h>
#include <structure/funcs/tensorRT/trt_executor.h>
#include <structure/graph_exec.h>
#include <maro/transform.h>
#include <structure/tensor.h>

int main() {
    mariana::ConvertContext ccontext;
    ccontext.ishapes.insert({"Conv_0", {1, 3, 640, 640}});
    ccontext.model_path = "serialize_engine_output.plan";
    ccontext.back_end   = mariana::Backend::TRT;
    mariana::Graph* graph = mariana::parse(ccontext);
    mariana::Tensor t(mariana::DeviceType::CPU);
    t.set_shape({1, 3, 640, 640});
    float* ptr = t.mutable_data<float>();
    // std::cout<<"d:"<<t.device()<<" "<<ptr<<"\n";
    // mariana::Tensor t1 = t.cpu();
    // float* ptr1 = t1.mutable_data<float>();
    // std::cout<<"d:"<<t1.device()<<" "<<ptr1<<"\n";
    mariana::ExecContext context;
    context.ishapes.insert({"images", {1, 3, 640, 640}});
    context.itensors.insert({"images", t});
    // std::cout<<"prerun"<<std::endl;
    
    mariana::GraphExec ge;
    ge.run(*graph, context);

    // mariana::transform::transform(graph, {"base_fold_reshape_to_node"});
    // mariana::trt::TensorRTEngine trt{};
    // trt.pre_run(graph, context);
    // std::cout<<"g:"<<graph<<std::endl;
    return 0;
}
