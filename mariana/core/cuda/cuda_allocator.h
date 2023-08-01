/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cuda/cuda_allocator.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-01:15:54:40
 * Description:
 *
 */

#ifndef __CORE_CUDA_CUDA_ALLOCATOR_H__
#define __CORE_CUDA_CUDA_ALLOCATOR_H__

#include <core/allocator.h>

namespace mariana {

MemoryReporter& cuda_memory_reporter();

Allocator* get_cuda_allocator();

void set_cuda_allocator(Allocator* alloc, uint8_t priority = 0);

} // namespace mariana

#endif /* __CORE_CUDA_CUDA_ALLOCATOR_H__ */

