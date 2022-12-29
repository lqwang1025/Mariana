/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/function.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-15:10:12:09
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCTION_H__
#define __STRUCTURE_FUNCTION_H__

#include <vector>
#include <functional>
#include <unordered_map>

#include <core/impl/type.h>
#include <structure/edge.h>
#include <structure/tensor.h>

namespace mariana {

using tensor_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

struct Function {
    Function() : next_(nullptr) {}
    virtual ~Function() = default;
    tensor_list operator()(tensor_list&& inputs) {
        return compute(std::move(inputs));
    }
    void set_next(std::shared_ptr<Function> next) {
        next_ = next;
    }
    virtual tensor_list compute(tensor_list&& inputs)=0;
protected:
    std::shared_ptr<Function> next_;
};

class FunctionHolder final {
public:
    using FuncMake = std::function<Function*()>;
    typedef std::unordered_map<OpCategory, FuncMake> FuncMap;
    static FuncMap& get_func_map() {
        static FuncMap* func_map = new FuncMap;
        return *func_map;
    }
    
    static void add_func(const OpCategory& category, FuncMake func) {
        FuncMap& func_map = get_func_map();
        if (func_map.count(category) == 1) {
            MVLOG(WARNING)<<"FUNC "<<category<<" had been registred.";
            return;
        }
        func_map[category] = func;
    }

    static FuncMake search(const OpCategory& category) {
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
    FunctionHolder()=delete;
    FunctionHolder(const FunctionHolder&)=delete;
    FunctionHolder& operator=(const FunctionHolder&)=delete;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCTION_H__ */

