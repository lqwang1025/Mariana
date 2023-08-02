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

namespace mariana {

using tensor_list = std::vector<Tensor>;
using result_list = std::vector<MResult>;

struct Processor {
    Processor() {}
    Processor(const ConvertContext&) {}
    virtual ~Processor() {}
    result_list operator()(tensor_list&& inputs, ExecContext& context) {
        result_list results = work(std::move(inputs), context);
        return results;
    }
    virtual result_list work(tensor_list&& inputs, ExecContext& context)=0;
};

class ProcessorHolder final {
public:
    using FuncMake = std::function<Processor*(const ConvertContext&)>;
    typedef std::unordered_map<ProcessorCategory, FuncMake> FuncMap;
    static FuncMap& get_func_map() {
        static FuncMap* func_map = new FuncMap;
        return *func_map;
    }
    
    static void add_func(const ProcessorCategory& category, FuncMake func) {
        FuncMap& func_map = get_func_map();
        if (func_map.count(category) == 1) {
            MVLOG(WARNING)<<"FUNC "<<static_cast<int>(category)<<" had been registred.";
            return;
        }
        func_map[category] = func;
    }

    static FuncMake search(const ProcessorCategory& category) {
        FuncMap& func_map = get_func_map();
        if (func_map.size() == 0 || func_map.count(category) == 0) {
            MVLOG(FATAL)<<"There is no func in registry:"<<static_cast<int>(category);
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

