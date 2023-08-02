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
#include <api/mariana_api.h>

namespace mariana {

class Graph;

Graph* parse(const ConvertContext& context);

} // namespace mariana

#endif /* __MARC_MARC_H__ */

