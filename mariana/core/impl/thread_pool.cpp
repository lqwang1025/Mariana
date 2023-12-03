/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/thread_pool.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-23:14:39:12
 * Description:
 * 
 */

#include <core/impl/thread_pool.h>

namespace mariana {

ThreadPool::ThreadPool(int pool_size, std::function<void()> init_thread)
    : shutdown_(false) , complete_(true)
    , threads_(pool_size < 0 ? default_num_threads() : pool_size)
    , total_(threads_.size()), available_(total_) {
    for (std::size_t i = 0; i < total_; ++i) {
        threads_[i] = std::thread([this, i, init_thread] {
                if (init_thread) init_thread();
                ThreadWorker work(this, i);
                work();
            });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
        condition_.notify_all();
    }
    for (auto& t : threads_) {
        try {
            t.join();
        } catch (const std::exception&) {}
    }
}

void ThreadPool::wait_work_complete() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (false == complete_) {
        completed_.wait(lock);
    }
}

void ThreadPool::ThreadWorker::operator()() {
    std::unique_lock<std::mutex> lock(thread_pool_->mutex_);
    while (false == thread_pool_->shutdown_) {
        // Wait on condition variable while the task is empty and 
        // the pool is still running.
        while (thread_pool_->tasks_.empty() && false == thread_pool_->shutdown_) {
            thread_pool_->condition_.wait(lock);
        }
        // If pool is no longer running, break out of loop.
        if (true == thread_pool_->shutdown_) break;
        {
            std::function<void()> task = std::move(thread_pool_->tasks_.front());
            thread_pool_->tasks_.pop();
            --thread_pool_->available_;
            lock.unlock();
            task();
        }
        lock.lock();
        ++thread_pool_->available_;
        if (thread_pool_->tasks_.empty() && thread_pool_->available_ == thread_pool_->total_) {
            thread_pool_->complete_ = true;
            thread_pool_->completed_.notify_one();
        }
    }
}

} // namespace mariana
