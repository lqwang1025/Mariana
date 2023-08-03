/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/ops.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:10:22:40
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_OPS_H__
#define __STRUCTURE_FUNCS_OPS_H__

#include <set>
#include <string>

namespace mariana {

#define MCONV2D "CONV2D"
#define MSOFTMAX "SOFTMAX"
#define MRESHAPE "RESHAPE"
#define MMAXPOOL "MAXPOOL"
#define MGAVPOOL "GAVPOOL"
#define MGEMM "GEMM"
#define MFLATTEN "FLATTEN"
#define MSPLIT "SPLIT"
#define MCONCAT "CONCAT"
#define MRESIZE "RESIZE"
#define MPERMUTE "PERMUTE"
#define MSLICE "SLICE"

//Math op
#define MADD "ADD"
#define MMUL "MUL"
#define MSUB "SUB"
#define MDIV "DIV"

// Activation op
#define MRELU "RELU"
#define MSIGMOID "SIGMOID"
    
static std::set<std::string> ACTIVATION_OP = {MRELU, MSIGMOID};

} // namespace mariana


#endif /* __STRUCTURE_FUNCS_OPS_H__ */

