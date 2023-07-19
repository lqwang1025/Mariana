/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_helper.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:18:01:10
 * Description:
 * 
 */

#include <structure/funcs/tensorRT/trt_helper.h>

namespace mariana { namespace trt {

constexpr long double operator"" _GiB(long double val) {
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val) {
    return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val) {
    return val * (1 << 10);
}

constexpr long long int operator"" _GiB(long long unsigned int val) {
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val) {
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val) {
    return val * (1 << 10);
}

}} // namespace mariana::trt
