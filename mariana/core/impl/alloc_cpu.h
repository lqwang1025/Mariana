/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/alloc_cpu.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:14:03:05
 * Description:
 *
 */

#ifndef __CORE_IMPL_ALLOC_CPU_H__
#define __CORE_IMPL_ALLOC_CPU_H__

namespace mariana {

void* alloc_cpu(size_t nbytes);

void free_cpu(void* data);

} // namespace mariana

#endif /* __CORE_IMPL_ALLOC_CPU_H__ */

