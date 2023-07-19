/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/device.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:10:07:51
 * Description:
 * 
 */

#include <core/device.h>

namespace mariana {

std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << device.type();
    return stream;
}

} // namespace mariana
