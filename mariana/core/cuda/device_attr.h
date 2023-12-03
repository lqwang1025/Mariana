/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/cuda/device_attr.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-24:09:11:30
 * Description:
 *
 */

#ifndef __CORE_CUDA_DEVICE_ATTR_H__
#define __CORE_CUDA_DEVICE_ATTR_H__

#include <cuda_runtime.h>

namespace mariana {

class CudaAttr {
public:
    CudaAttr();
    int device_count() const;
private:
    int device_count_;
};


} // namespace mariana

#endif /* __CORE_CUDA_DEVICE_ATTR_H__ */

