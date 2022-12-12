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
 
int main() {
    mariana::TensorImpl t;
    std::cout<<"debug:"<<t.device()<<std::endl;
    float* s = t.mutable_data<float>();
    return 0;
}

