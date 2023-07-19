/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/tensorRT/trt_helper.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:17:57:35
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_TENSORRT_TRT_HELPER_H__
#define __STRUCTURE_FUNCS_TENSORRT_TRT_HELPER_H__

namespace mariana { namespace trt {

constexpr long double operator"" _GiB(long double val);

constexpr long double operator"" _MiB(long double val);

constexpr long double operator"" _KiB(long double val);

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val);

constexpr long long int operator"" _MiB(long long unsigned int val);

constexpr long long int operator"" _KiB(long long unsigned int val);


}} // namespace mariana::trt

#endif /* __STRUCTURE_FUNCS_TENSORRT_TRT_HELPER_H__ */

