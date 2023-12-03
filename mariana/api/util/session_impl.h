/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : api/util/session_impl.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-25:13:48:15
 * Description:
 *
 */

#ifndef __API_UTIL_SESSION_IMPL_H__
#define __API_UTIL_SESSION_IMPL_H__

#include <memory>
#include <queue>
#include <mutex>
#include <string>
#include <unordered_map>
#include <condition_variable>
#include <api/mariana_api.h>
#include <core/impl/thread_pool.h>

namespace mariana {

class RuntimeImpl {
public:
    explicit RuntimeImpl(int max_isize, const std::string& prototxt,
                         int concurrency_size);
    ~RuntimeImpl();
    void push(ExecContext& context);
    void run(Runtime runtime);
    MResult pull(ExecContext& context);
    void shutdown();
    void add_obuffer(const std::string& buffer_name);
    std::vector<std::string> inames() const {
        return input_names_;
    }
    std::vector<std::string> onames() const {
        return input_names_;
    }
private:
    std::atomic_int max_ibuffer_size_;
    std::atomic_bool shutdown_;
    std::string prototxt_;
    mutable std::mutex imutex_;
    mutable std::mutex omutex_;
    std::condition_variable pushcv_;
    std::condition_variable runcv_;
    std::condition_variable ocv_;
    std::queue<ExecContext> ibuffer_;
    std::unordered_map<std::string, std::unordered_map<std::string, MResult>> obuffer_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

class SessionImpl {
public:
    SessionImpl() {}
    ~SessionImpl() {}
    RuntimeImpl* add_runtime(const char* prototxt, const ScheduleConfig& config);
private:
    std::vector<std::unique_ptr<RuntimeImpl>> runtimes_;
};

} // namespace mariana

#endif /* __API_UTIL_SESSION_IMPL_H__ */

