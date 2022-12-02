/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/device_type.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:09:44:06
 * Description:
 * 
 */

#include <ctype.h>

#include <core/device_type.h>
#include <core/utils/logging.h>

namespace mariana {

std::string device_type_name(DeviceType d, bool lower_case) {
    switch (d) {
    case DeviceType::CPU:
        return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
        return lower_case ? "cuda" : "CUDA";
    case DeviceType::FPGA:
        return lower_case ? "fpga" : "FPGA";
    default:
        MCHECK(false)<<"Unknown device: "<<static_cast<int16_t>(d);
        // The below code won't run but is needed to suppress some compiler
        // warnings.
        return "";
    }
}

bool is_valid_device_type(DeviceType d) {
    switch (d) {
    case DeviceType::CPU:
    case DeviceType::CUDA:
    case DeviceType::FPGA:
        return true;
    default:
        return false;
    }
}

std::ostream& operator<<(std::ostream& stream, DeviceType type) {
    stream << device_type_name(type, /* lower case */true);
    return stream;
}

} // namespace mariana
