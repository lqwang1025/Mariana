/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/fetch.h
 * Authors    : wangliquan@zkln
 * Create Time: 2023-01-04:10:22:01
 * Description:
 *
 */

#ifndef __STRUCTURE_FETCH_H__
#define __STRUCTURE_FETCH_H__

namespace mariana {

class Fetch {
    Fetch() {}
    virtual ~Fetch() {}
    void before_using_input();
};

} // namespace mariana

#endif /* __STRUCTURE_FETCH_H__ */

