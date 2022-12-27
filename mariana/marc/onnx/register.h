/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : marc/onnx/register.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:46:21
 * Description:
 *
 */

#ifndef __MARC_ONNX_REGISTER_H__
#define __MARC_ONNX_REGISTER_H__

#include <marc/onnx/onnx.h>

namespace mariana { namespace onnx {

void register_converter();

void unregister_converter();

DECLARE_ONNX_CONVERTER_CLASS(AddConverter);
DECLARE_ONNX_CONVERTER_CLASS(ConcatConverter);
DECLARE_ONNX_CONVERTER_CLASS(ConvConverter);
DECLARE_ONNX_CONVERTER_CLASS(MaxPoolConverter);
DECLARE_ONNX_CONVERTER_CLASS(MulConverter);
DECLARE_ONNX_CONVERTER_CLASS(PowConverter);
DECLARE_ONNX_CONVERTER_CLASS(ReshapeConverter);
DECLARE_ONNX_CONVERTER_CLASS(ResizeConverter);
DECLARE_ONNX_CONVERTER_CLASS(SigmoidConverter);
DECLARE_ONNX_CONVERTER_CLASS(SplitConverter);
DECLARE_ONNX_CONVERTER_CLASS(TransposeConverter);
DECLARE_ONNX_CONVERTER_CLASS(ReluConverter);
DECLARE_ONNX_CONVERTER_CLASS(DefaultConverter);

}} // namespace mariana::onnx

#endif /* __MARC_ONNX_REGISTER_H__ */

