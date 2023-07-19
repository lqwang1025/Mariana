/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:10:55:51
 * Description:
 *
 */

#ifndef __MARC_ONNX_OPS_H__
#define __MARC_ONNX_OPS_H__

#include <set>
#include <string>

namespace mariana { namespace onnx {

#define KConv "Conv"
#define KRelu "Relu"
#define KSoftmax "Softmax"
#define KReshape "Reshape"
#define KMaxPool "MaxPool"
#define KAdd "Add"
#define KGlobalAveragePool "GlobalAveragePool"
#define KFlatten "Flatten"
#define KGemm "Gemm"

// since 13
#define KIdentity "Identity"

extern std::set<std::string> CONTINUE_OP;

}} // namespace mariana::onnx

#endif /* __MARC_ONNX_OPS_H__ */

