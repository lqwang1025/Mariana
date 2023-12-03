/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : api/util/io.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-24:16:50:14
 * Description:
 *
 */

#ifndef __API_UTIL_IO_H__
#define __API_UTIL_IO_H__

#include <google/protobuf/message.h>

namespace mariana {

bool read_proto_from_text_file(const char* filename, ::google::protobuf::Message* proto);

void write_proto_to_text_file(const ::google::protobuf::Message& proto, const char* filename);

} // namespace mariana

#endif /* __API_UTIL_IO_H__ */

