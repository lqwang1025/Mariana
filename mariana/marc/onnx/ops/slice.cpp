/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : slice.cpp
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-27:16:52:24
 * Description:
 * 
 */

#include <marc/onnx/register.h>
#include <structure/funcs/slice.h>
#include <marc/onnx/proto/onnx_help.h>
#include <core/utils/logging.h>
#include <core/utils/arrary_ref.h>

namespace mariana { namespace onnx {

void SliceConverter::run(const ::onnx::NodeProto& src, Node& dst, const OnnxScope& scope) {
    SliceFunction* func = static_cast<SliceFunction*>(dst.op());

    int32_t* begin = &func->option.begin;
    
    for (int i = 1; i < src.input_size(); ++i) {
        const ::onnx::TensorProto* attr = scope.graph_info.tensor_name_map.at(src.input(i));
        std::vector<int64_t> shape;
        void* content = nullptr;
        get_content_from_tensor(*attr, shape, &content);
        MCHECK(shape.size() == 1)<<"Wrong!";
        
        ::onnx::TensorProto_DataType data_type = static_cast<::onnx::TensorProto_DataType>(attr->data_type());
        switch (data_type) {
        case ::onnx::TensorProto_DataType::TensorProto_DataType_INT32 : {
            int32_t value = *static_cast<int32_t*>(content);
            *(begin+i-1) = static_cast<int32_t>(value);
            break;
        }
        case ::onnx::TensorProto_DataType::TensorProto_DataType_INT64 : {
            int64_t value = *static_cast<int64_t*>(content);
            *(begin+i-1) = static_cast<int32_t>(value);
            break;
        }
        default: {
            MLOG(FATAL)<<"Mar Fatal: unsupport data type in slice.";
        }            
        }
    }
}

}} // namespace mariana::onnx

