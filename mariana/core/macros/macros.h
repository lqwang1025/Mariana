/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/macros/macros.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:09:13:55
 * Description:
 *
 */

#ifndef __CORE_MACROS_MACROS_H__
#define __CORE_MACROS_MACROS_H__

#define MAR_DISABLE_COPY_AND_ASSIGN(classname)      \
    classname(const classname&) = delete;           \
    classname& operator=(const classname&) = delete

#define MAR_CONCATENATE_IMPL(s1, s2) s1##s2
#define MAR_CONCATENATE(s1, s2) C10_CONCATENATE_IMPL(s1, s2)

#define MAR_MACRO_EXPAND(args) args

#define MAR_STRINGIZE_IMPL(x) #x
#define MAR_STRINGIZE(x) MAR_STRINGIZE_IMPL(x)

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define MAR_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define MAR_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define MAR_LIKELY(expr) (expr)
#define MAR_UNLIKELY(expr) (expr)
#endif

#ifdef __GNUC__
#define MAR_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define MAR_NOINLINE __declspec(noinline)
#else
#define MAR_NOINLINE
#endif

#if defined(_MSC_VER)
#define MAR_ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define MAR_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define MAR_ALWAYS_INLINE inline
#endif

namespace mariana {}

#endif /* __CORE_MACROS_MACROS_H__ */

