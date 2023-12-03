/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : rknn_executor.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-04:11:23:10
 * Description:
 * 
 */

#include <cstdio>
#include <core/utils/logging.h>
#include <structure/funcs/rknn/rknn_executor.h>

#define RET_CHECKER(ret)											\
	do {															\
		if ((ret) < 0) {											\
            MLOG(ERROR)<<"Rknn ret code error[line]:"<<__LINE__;    \
		}															\
	} while(false)

namespace mariana { namespace rknn {

RknnEngine::RknnEngine() {
    
}

RknnEngine::~RknnEngine() {
	RET_CHECKER(rknn_destroy(ctx_));
}

uint8_t* RknnEngine::_load_data(FILE* fp, size_t ofst, size_t sz) {
    unsigned char* data;
	int            ret;
	data = NULL;
	if (NULL == fp) {
		return NULL;
	}
	ret = fseek(fp, ofst, SEEK_SET);
	if (ret != 0) {
		printf("blob seek failure.\n");
		return NULL;
	}
	data = (unsigned char*)malloc(sz);
	if (data == NULL) {
		printf("buffer malloc failure.\n");
		return NULL;
	}
	ret = fread(data, 1, sz, fp);
	return data;
}

uint8_t* RknnEngine::_load_model(const std::string& filename, int* model_size) {
    FILE*          fp;
	unsigned char* data;

	fp = fopen(filename.c_str(), "rb");
	if (NULL == fp) {
		MLOG(FATAL)<<"Open rknn model:"<<filename<<" failed.";
	}
	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);
	data = _load_data(fp, 0, size);
	fclose(fp);
	*model_size = size;
	return data;
}

Status RknnEngine::de_serialize(Graph& graph, const proto::ModelInfo& model_info) {
    uint8_t* model_data = _load_model(model_info.model_path(), &model_size_);
    RET_CHECKER(rknn_init(&ctx_, model_data, model_size_, 0));
    free(model_data);
    rknn_sdk_version version;
	RET_CHECKER(rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)));
    printf("RKNN version: %s Driver version: %s\n", version.api_version, version.drv_version);
    RET_CHECKER(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_)));
    input_attrs_.resize(io_num_.n_input);
    memset(input_attrs_.data(), 0, io_num_.n_input*sizeof(rknn_tensor_attr));
    rknn_inputs_.resize(io_num_.n_input);
    memset(rknn_inputs_.data(), 0, io_num_.n_input*sizeof(rknn_input));
    output_attrs_.resize(io_num_.n_output);
    memset(output_attrs_.data(), 0, io_num_.n_output*sizeof(rknn_tensor_attr));
    for (size_t i = 0; i < input_attrs_.size(); ++i) {
        input_attrs_[i].index = i;
        RET_CHECKER(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr)));
        Tensor tensor(DeviceType::CPU);
        std::vector<int64_t> shapes;
        for (int n = input_attrs_[i].n_dims-1; n >= 0; --n) {
            shapes.push_back(input_attrs_[i].dims[n]);
        }
        IntArrayRef shape(shapes);
        tensor.set_shape(shape);
        tensor.set_name(input_attrs_[i].name);
        itensors.push_back(tensor);
    }
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
        output_attrs_[i].index = i;
        RET_CHECKER(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr)));
        Tensor tensor(DeviceType::CPU);
        std::vector<int64_t> shapes;
        for (int n = output_attrs_[i].n_dims-1; n >= 0; --n) {
            shapes.push_back(output_attrs_[i].dims[n]);
        }
        
        IntArrayRef shape(shapes);
        tensor.set_shape(shape);
        tensor.set_name(output_attrs_[i].name);

        // todo : base on datatype
        tensor.mutable_data<float>();
        otensors.push_back(tensor);
    }
    return absl::OkStatus();
}

Status RknnEngine::run(const ExecContext& context) {
    // rknn_input inputs[itensors.size()];
    // memset(inputs, 0, sizeof(inputs));
    int index = 0;
    for (auto& tensor : itensors) {
        if (context.itensors.count(tensor.name()) == 0) {
            MLOG(FATAL)<<"Input name:"<<tensor.name()<<" do not in input tensors";
        } else {
            void*             input    = context.itensors.at(tensor.name()).input;
            const TypeMeta&   dtype    = context.itensors.at(tensor.name()).dtype;
            if (dtype.match<uint8_t>()) {
                rknn_inputs_[index].type         = RKNN_TENSOR_UINT8;
            } else if (dtype.match<float>()) {
                rknn_inputs_[index].type         = RKNN_TENSOR_FLOAT32;
            } else {
                MLOG(FATAL)<<"[rknn] Upsupport data type :"<<dtype.data().name;
            }
                 
			rknn_inputs_[index].size         = tensor.numel();
			rknn_inputs_[index].fmt          = RKNN_TENSOR_NHWC;
			rknn_inputs_[index].pass_through = 0;
			rknn_inputs_[index].buf          = input;
            rknn_inputs_[index].index        = index;
            index++;
        }
    }
    RET_CHECKER(rknn_inputs_set(ctx_, io_num_.n_input, rknn_inputs_.data()));

    rknn_output outputs[io_num_.n_output];
    for (int i = 0; i < io_num_.n_output; i++) {
        outputs[i].index       = i;
        outputs[i].want_float  = 1;
        outputs[i].is_prealloc = 1;
        outputs[i].size        = otensors[i].numel()*otensors[i].itemsize();
        outputs[i].buf         = otensors[i].data();
    }
    RET_CHECKER(rknn_run(ctx_, NULL));
    RET_CHECKER(rknn_outputs_get(ctx_, io_num_.n_output, outputs, NULL));
    RET_CHECKER(rknn_outputs_release(ctx_, io_num_.n_output, outputs));
    
    return absl::OkStatus();
}

}} // namespace mariana::rknn
