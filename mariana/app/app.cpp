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
#include <core/allocator.h>
#include <core/tensor_impl.h>
#include <core/impl/shape.h>

int main(int argc, char** argv) {
    mariana::allocator_context().fill_junk = true;
    mariana::allocator_context().report_memory = true;
    mariana::Allocator* alloc = mariana::get_allocator(mariana::DeviceType::CPU);
    auto dataptr = alloc->allocate(32*sizeof(float));
    float* data = static_cast<float*>(dataptr.get());
    for (int i = 0; i < 32; ++i) {
        std::cout<<"debug:"<<data[i]<<std::endl;
    }
    std::cout<<"debug:"<<dataptr.get()<<" "<<&dataptr<<std::endl;
    alloc->mdelete(dataptr);
    std::cout<<"debug:"<<dataptr.get()<<std::endl;
    return 0;
}

