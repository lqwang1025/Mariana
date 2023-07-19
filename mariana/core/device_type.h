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

namespace mariana {

enum class DeviceType : int8_t {
    UNINIT=0,
    CPU=0,
    CUDA=1,
    FPGA=2,
    // If you add other devices
    //  - Change the implementations of DeviceTypeName and isValidDeviceType
    //    in device_ype.cpp
    //  - Change the number below
    COMPILE_TIME_MAX_DEVICE_TYPES=3
};

    
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

