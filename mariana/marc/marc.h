/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/marc.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-26:17:40:30
 * Description:
 *
 */

#ifndef __MARC_MARC_H__
#define __MARC_MARC_H__

#include <string>
#include <unordered_map>

#include <core/impl/shape.h>

namespace mariana {

class Graph;

enum class Backend : int8_t {
    UNINIT = -1,
    RKNN   = 0,
    TRT    = 1,
};

enum class ModelMode : int8_t {
    UNINIT  = -1,
    FP16    = 0,
    FP32    = 0,
    INT8    = 2,
    QATINT8 = 3,
};

struct ConvertContext {
    ConvertContext() {}
    ~ConvertContext() {}
    std::unordered_map<std::string, Shape> ishapes;
    std::string model_path;
    bool from_scratch = false;
    Backend back_end = Backend::UNINIT;
    int max_batch_size = 1;
    ModelMode mode = ModelMode::FP16;
};

Graph* parse(const ConvertContext& context);

} // namespace mariana

#endif /* __MARC_MARC_H__ */

