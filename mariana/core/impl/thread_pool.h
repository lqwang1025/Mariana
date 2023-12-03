/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/thread_pool.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-23:14:36:35
 * Description:
 *
 */

#ifndef __CORE_IMPL_THREAD_POOL_H__
#define __CORE_IMPL_THREAD_POOL_H__

#include <atomic>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>

namespace mariana {

class ThreadPool {
    class ThreadWorker;
public:
     explicit ThreadPool(int pool_size, std::function<void()> init_thread=nullptr);
    ~ThreadPool();

    size_t size() const {
        return threads_.size();
    }

    size_t num_available() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return available_;
    }

    void wait_work_complete();
    
    static size_t default_num_threads() {
        auto num_threads = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
        num_threads /= 2;
#endif
        return num_threads;
    }
    
    template<typename F, typename...Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        if (threads_.size() == 0) {
            throw std::runtime_error("No threads to run a task");
        }
        std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);
        std::function<void()> wrapper_func = [task_ptr]() {
            (*task_ptr)(); 
        };
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.emplace(wrapper_func);
        
        complete_ = false;
        condition_.notify_one();
        return task_ptr->get_future();
    }

private:
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    
    ThreadPool & operator=(const ThreadPool &) = delete;
    ThreadPool & operator=(ThreadPool &&) = delete;
    class ThreadWorker {
    public:
        ThreadWorker(ThreadPool* pool, const int id)
            : id_(id), thread_pool_(pool) {}
        
        void operator()();
    private:
        int id_;
        ThreadPool* thread_pool_;
    }; // class ThreadWorker

    std::atomic_bool shutdown_;
    bool complete_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> threads_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completed_;
    std::size_t total_;
    std::size_t available_;
};

} // namespace mariana

#endif /* __CORE_IMPL_THREAD_POOL_H__ */

