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
#include <core/tensor_impl.h>
#include <core/layout.h>
#include <core/device.h>
#include <core/utils/typemeta.h>
 
int main() {
    // mariana::TensorImpl t;
    // t.set_shape({1,3,224, 224});
    // std::cout<<"debug:"<<t.shape()<<" "<<t.stride()<<std::endl;
    // float* s = t.mutable_data<float>();
    std::cout<<"layout:"<<sizeof(mariana::Layout)<<std::endl;
    std::cout<<"device:"<<sizeof(mariana::Device)<<std::endl;
    std::cout<<"TypeMeta:"<<sizeof(mariana::TypeMeta)<<std::endl;
    
    return 0;
}

