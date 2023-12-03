/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/processor.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-02:13:26:30
 * Description:
 *
 */

#ifndef __STRUCTURE_PROCESSOR_H__
#define __STRUCTURE_PROCESSOR_H__

#include <vector>
#include <functional>
#include <unordered_map>

#include <core/utils/logging.h>
#include <structure/tensor.h>
#include <api/mariana_api.h>
#include <api/proto/mariana.pb.h>

namespace mariana {

using tensor_list = std::vector<Tensor>;

struct Processor {
    Processor() {}
    Processor(const proto::ModelInfo& model_info) {} // for register
    virtual ~Processor() {}
    MResult operator()(tensor_list&& inputs, ExecContext& context) {
        MResult results = work(std::move(inputs), context);
        return results;
    }
    virtual MResult work(tensor_list&& inputs, ExecContext& context)=0;
};

class ProcessorHolder final {
public:
    using FuncMake = std::function<Processor*(const proto::ModelInfo&)>;
    typedef std::unordered_map<proto::PostProcessorCategory, FuncMake> FuncMap;
    static FuncMap& get_func_map() {
        static FuncMap* func_map = new FuncMap;
        return *func_map;
    }
    
    static void add_func(const proto::PostProcessorCategory& category, FuncMake func) {
        FuncMap& func_map = get_func_map();
        if (func_map.count(category) == 1) {
            MVLOG(WARNING)<<"FUNC "<<category<<" had been registred.";
            return;
        }
        func_map[category] = func;
    }

    static FuncMake search(const proto::PostProcessorCategory& category) {
        FuncMap& func_map = get_func_map();
        if (func_map.size() == 0 || func_map.count(category) == 0) {
            MVLOG(FATAL)<<"There is no func in registry:"<<category;
            return nullptr;
        }
        return func_map[category];
    }
    
    static void release() {
        FuncMap& func_map = get_func_map();
        func_map.clear();
    }
private:
    ProcessorHolder()=delete;
    ProcessorHolder(const ProcessorHolder&)=delete;
    ProcessorHolder& operator=(const ProcessorHolder&)=delete;
};

} // namespace mariana

#endif /* __STRUCTURE_PROCESSOR_H__ */

