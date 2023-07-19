/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cpu_allocator.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:13:36:22
 * Description:
 *
 */

#ifndef __CORE_CPU_ALLOCATOR_H__
#define __CORE_CPU_ALLOCATOR_H__

#include <core/allocator.h>

namespace mariana {

MemoryReporter& cpu_memory_reporter();

Allocator* get_cpu_allocator();

void set_cpu_allocator(Allocator* alloc, uint8_t priority = 0);

} // namespace mariana

#endif /* __CORE_CPU_ALLOCATOR_H__ */

