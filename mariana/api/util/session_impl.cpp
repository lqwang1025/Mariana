/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : api/util/session_impl.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-25:13:48:42
 * Description:
 * 
 */

#include <iostream>
#include <api/util/session_impl.h>

namespace mariana {

void RuntimeImpl::shutdown() {
    shutdown_ = true;
    ocv_.notify_all();
    pushcv_.notify_all();
    runcv_.notify_all();
}

RuntimeImpl::~RuntimeImpl() {
    thread_pool_.release();
    shutdown();
}

void RuntimeImpl::add_obuffer(const std::string& buffer_name) {
    obuffer_.insert({buffer_name, {}});
}

RuntimeImpl::RuntimeImpl(int max_isize, const std::string& prototxt,
                         int concurrency_size)
    : max_ibuffer_size_(max_isize), shutdown_(false), prototxt_(prototxt) {
    thread_pool_ = std::unique_ptr<ThreadPool>(new ThreadPool{concurrency_size});
    for (int i = 0; i < concurrency_size; ++i) {
        Runtime runtime = Runtime::create_from(prototxt_.c_str());
        if (input_names_.empty()) {
            input_names_ = runtime.input_names;
        }
        if (output_names_.empty()) {
            output_names_ = runtime.output_names;
        }
        thread_pool_->submit(std::mem_fn(&RuntimeImpl::run), this, runtime);
    }
}

void RuntimeImpl::push(ExecContext& context) {
    std::unique_lock<std::mutex> lock(imutex_);
    while (max_ibuffer_size_ <= ibuffer_.size() && false == shutdown_) {
        pushcv_.wait(lock);
    }
    if (true == shutdown_) return;
    ibuffer_.emplace(context);
    lock.unlock();
    runcv_.notify_one();
}

MResult RuntimeImpl::pull(ExecContext& context) {
    std::unique_lock<std::mutex> lock(omutex_);
    while (obuffer_[context.channel_id].count(context.identification) == 0 && false == shutdown_) {
        ocv_.wait(lock);
    }
    MResult results = obuffer_[context.channel_id][context.identification];
    obuffer_[context.channel_id].erase(context.identification);
    lock.unlock();
    return results;
}

void RuntimeImpl::run(Runtime runtime) {
    std::unique_lock<std::mutex> ilock(imutex_);
    while (false == shutdown_) {
        while (true == ibuffer_.empty() && false == shutdown_) {
            runcv_.wait(ilock);
        }
        if (true == shutdown_) break;
        ExecContext context = ibuffer_.front();
        ibuffer_.pop();
        ilock.unlock();
        pushcv_.notify_all();
        {
            MResult results = runtime.run_with(context);
            std::unique_lock<std::mutex> olock(omutex_);
            obuffer_[context.channel_id][context.identification] = results;
        }
        ocv_.notify_all();
        ilock.lock();
    }
    runtime.destory();
}

RuntimeImpl* SessionImpl::add_runtime(const char* prototxt, const ScheduleConfig& config) {
    RuntimeImpl* runtime_impl = new RuntimeImpl{config.max_ibuffer_size,
                                                std::string(prototxt),
                                                config.num_thread};
    runtimes_.push_back(std::unique_ptr<RuntimeImpl>(runtime_impl));
    return runtime_impl;
}

} // namespace mariana
