/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cuda/alloc_cuda.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-01:15:35:55
 * Description:
 *
 */

#ifndef __CORE_CUDA_ALLOC_CUDA_H__
#define __CORE_CUDA_ALLOC_CUDA_H__

namespace mariana {

void* alloc_cuda(size_t nbytes);

void free_cuda(void* data);

} // namespace mariana


#endif /* __CORE_CUDA_ALLOC_CUDA_H__ */

