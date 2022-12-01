/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/utils/sys.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:09:38:32
 * Description:
 *
 */

#ifndef __CORE_UTILS_SYS_H__
#define __CORE_UTILS_SYS_H__

#include <core/utils/status.h>

namespace mariana {

Status create_folders(const char *dir);

} // namespace mariana

#endif /* __CORE_UTILS_SYS_H__ */

