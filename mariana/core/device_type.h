/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/device_type.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:09:35:33
 * Description:
 *
 */

#ifndef __CORE_DEVICE_TYPE_H__
#define __CORE_DEVICE_TYPE_H__

#include <string>
#include <ostream>

#include <api/mariana_api.h>

namespace mariana {
    
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kFPGA = DeviceType::FPGA;

constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

std::string device_type_name(DeviceType d, bool lower_case = false);

bool is_valid_device_type(DeviceType d);

std::ostream& operator<<(std::ostream& stream, DeviceType type);

} // namespace mariana

#endif /* __CORE_DEVICE_TYPE_H__ */

