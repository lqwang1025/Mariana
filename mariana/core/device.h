/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/device.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:09:32:09
 * Description:
 *
 */

#ifndef __CORE_DEVICE_H__
#define __CORE_DEVICE_H__

#include <string>
#include <core/device_type.h>

namespace mariana {

struct Device final {
    Device(DeviceType type) : type_(type) {}
    bool operator==(const Device& other) const {
        return type_ == other.type_;
    }
    bool operator!=(const Device& other) const {
        return type_ != other.type_;
    }
    bool is_cpu() const {
        return type_ == DeviceType::CPU;
    }
    bool is_cuda() const {
        return type_ == DeviceType::CUDA;
    }
    bool is_fpga() const {
        return type_ == DeviceType::FPGA;
    }
    DeviceType type() const noexcept {
        return type_;
    }
private:
    DeviceType type_;
};

std::ostream& operator<<(std::ostream& stream, const Device& device);

} // namespace mariana

#endif /* __CORE_DEVICE_H__ */

