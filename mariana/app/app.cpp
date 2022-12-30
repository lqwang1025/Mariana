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
#include <core/allocator.h>
#include <core/cpu_allocator.h>

int main() {
    // mariana::TensorImpl t;
    // t.set_shape({1,3,224, 224});
    // std::cout<<"debug:"<<t.shape()<<" "<<t.stride()<<std::endl;
    // float* s = t.mutable_data<float>();
    
    mariana::parse("/home/wangliquan/learn/Mariana/mariana/build/res.onnx");
    // mariana::get_cpu_allocator()->allocate(1024);
    return 0;
}

