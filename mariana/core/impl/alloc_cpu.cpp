/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/alloc_cpu.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-02:14:04:06
 * Description:
 * 
 */

#include <core/alignment.h>
#include <core/utils/logging.h>
#include <core/allocator.h>
#include <core/impl/alloc_cpu.h>

namespace mariana {

namespace {

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
void memset_junk(void* data, size_t num) {
    // This garbage pattern is NaN when interpreted as floating point values,
    // or as very large integer values.
    static constexpr int32_t kJunkPattern = 0x7fedbeef;
    static constexpr int64_t kJunkPattern64 =
        static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
    int32_t int64_count = num / sizeof(kJunkPattern64);
    int32_t remaining_bytes = num % sizeof(kJunkPattern64);
    int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
    for (int i = 0; i < int64_count; ++i) {
        data_i64[i] = kJunkPattern64;
    }
    if (remaining_bytes > 0) {
        memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
    }
} // memset_junk
} // namespace

void* alloc_cpu(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }
    MCHECK_GT(nbytes, 0)<<"alloc_cpu() seems to have been "
                        <<"called with negative number: "<<nbytes;
    void* data;

    int err = posix_memalign(&data, gAlignment, nbytes);
    MCHECK(err==0)<<"DefaultCPUAllocator: can't allocate memory: you tried to allocate "
                  <<nbytes<<" bytes. Error code "<<err<<" ("<<strerror(err)<<")";
    if (allocator_context().fill_zero) {
        memset(data, 0, nbytes);
    } else if (allocator_context().fill_junk) {
        memset_junk(data, nbytes);
    }
    return data;
}

void free_cpu(void* data) {
    free(data);
}

} // namespace mariana
