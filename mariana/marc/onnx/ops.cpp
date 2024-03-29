/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/ops.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:04:55
 * Description:
 * 
 */

#include <marc/onnx/ops.h>
#include <structure/funcs/ops.h>

namespace mariana { namespace onnx {

std::set<std::string> CONTINUE_OP = {
    KIdentity
};
std::map<std::string, std::string> ONNX_OP_MAP_TO_MAR = {
    {KConv, MCONV2D}, 
    {KRelu, MRELU},
    {KSoftmax, MSOFTMAX},
    {KReshape, MRESHAPE},
    {KMaxPool, MMAXPOOL},
    {KAdd, MADD},
    {KMul, MMUL},
    {KDiv, MDIV},
    {KSub, MSUB},
    {KGlobalAveragePool, MGAVPOOL},
    {KFlatten, MFLATTEN},
    {KGemm, MGEMM},
    {KSigmoid, MSIGMOID},
    {KSplit, MSPLIT},
    {KSlice, MSLICE},
    {KConcat, MCONCAT},
    {KResize, MRESIZE},
    {KTranspose, MPERMUTE},
    {kReduceMean, MREDUCEMEAN},
};

}} // namespace mariana::onnx
