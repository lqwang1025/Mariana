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
#include <core/tensor.h>
#include <core/impl/shape.h>

int main(int argc, char** argv) {
    mariana::Shape s = {1,3,244,244};
    std::cout<<"debug:"<<s.stride_at(0)<<std::endl;
    std::cout<<"debug:"<<s.stride_at(1)<<std::endl;
    std::cout<<"debug:"<<s.stride_at(2)<<std::endl;
    std::cout<<"debug:"<<s.stride_at(3)<<std::endl;
    return 0;
}
