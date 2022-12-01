/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/utils/sys.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:09:40:10
 * Description:
 * 
 */

#include <cstdlib>
#include <string>
#include <core/utils/sys.h>
#include <core/utils/logging.h>

namespace mariana {

Status create_folders(const char *dir) {
    char order[100] = "mkdir -p ";
    strcat(order, dir);
    int ret = system(order);
    if (ret==-1) {
        std::string ret = std::string("Create ")+dir+std::string(" failed.");
        return absl::InternalError(ret);
    }
    return absl::OkStatus();
}

} // namespace mariana
