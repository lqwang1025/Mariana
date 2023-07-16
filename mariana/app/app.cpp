/*
 *        (C) COPYRIGHT LeiNao Limited.
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
#include <maro/transform_utils.h>

int main() {
    // mariana::TensorImpl t;
    // t.set_shape({1,3,224, 224});
    // std::cout<<"debug:"<<t.shape()<<" "<<t.stride()<<std::endl;
    // float* s = t.mutable_data<float>();
    // mariana::get_cpu_allocator()->allocate(1024);
    mariana::Graph* graph = mariana::parse("/home/home/lqwang/project/Mariana/mariana/build/res.onnx");
    mariana::ExecContext context;
    context.ishapes.insert({"Conv_0", {1, 3, 224, 224}});

    std::cout<<"prerun"<<std::endl;
    
    mariana::GraphExec ge;
    ge.pre_run(*graph, context);

    std::cout<<"optttttttt"<<std::endl;
    mariana::transform::TransformRegistry* opt = mariana::transform::get_transform_registry();
    (*opt)["base_fold_reshape_to_node"](*graph);
    mariana::trt::TensorRTEngine trt{};
    trt.pre_run(*graph, context);
    return 0;
}
