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

// #include <iostream>
// #include <thread>
// #include <core/allocator.h>
// #include <core/tensor_impl.h>
// #include <core/impl/shape.h>

// void func() {
//     mariana::Allocator* alloc = mariana::get_allocator(mariana::DeviceType::CPU);
//     auto dataptr = alloc->allocate(32*sizeof(float));
//     float* data = static_cast<float*>(dataptr.get());
//     for (int i = 0; i < 32; ++i) {
//         std::cout<<"debug:"<<data[i]<<std::endl;
//     }
//     std::cout<<"debug:"<<dataptr.get()<<" "<<&dataptr<<std::endl;
//     alloc->mdelete(dataptr);
//     std::cout<<"debug:"<<dataptr.get()<<std::endl;
// }

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
 
struct Base
{
    Base() { std::cout << "  Base::Base()\n"; }
    // Note: non-virtual destructor is OK here
    ~Base() { std::cout << "  Base::~Base()\n"; }
};
 
struct Derived: public Base
{
    Derived() { std::cout << "  Derived::Derived()\n"; }
    ~Derived() { std::cout << "  Derived::~Derived()\n"; }
};
 
void thr(std::shared_ptr<Base> p)
{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<Base> lp = p; // thread-safe, even though the
                                  // shared use_count is incremented
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout << "local pointer in a thread:\n"
                  << "  lp.get() = " << lp.get()
                  << ", lp.use_count() = " << lp.use_count() << '\n';
    }
}
 
int main() {
    std::shared_ptr<Base> p = std::make_shared<Derived>();
    std::cout<<"debug:"<<p.use_count()<<std::endl;
    std::shared_ptr<Base> s = p;
    std::cout<<"debug:"<<s.use_count()<<std::endl;
    std::cout<<"debug:"<<(p==s)<<std::endl;
}

// int main(int argc, char** argv) {
    // mariana::allocator_context().fill_junk = true;
    // mariana::allocator_context().report_memory = true;
    // std::thread t(func);
    // mariana::allocator_context().fill_junk = false;
    // mariana::allocator_context().report_memory = false;
    // std::thread t1(func);
    // t.join();
    // t1.join();
    
//     return 0;
// }

