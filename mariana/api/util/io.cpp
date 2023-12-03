/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : io.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-24:16:50:18
 * Description:
 * 
 */
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <api/util/io.h>
#include <core/utils/logging.h>

namespace mariana {

bool read_proto_from_text_file(const char* filename, ::google::protobuf::Message* proto) {
    int fd = open(filename, O_RDONLY);
    MLOG_IF(ERROR, fd==-1)<<"File not found: "<<filename;
    if (fd == -1) return false;
        
    ::google::protobuf::io::FileInputStream* input = new ::google::protobuf::io::FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
}

void write_proto_to_text_file(const ::google::protobuf::Message& proto, const char* filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    ::google::protobuf::io::FileOutputStream* output = new ::google::protobuf::io::FileOutputStream(fd);
    if(google::protobuf::TextFormat::Print(proto, output) == false) {
        delete output;
        close(fd);
        MLOG(ERROR)<<"Write File failed: "<<filename;
    }
    delete output;
    close(fd);
}

} // namespace mariana
