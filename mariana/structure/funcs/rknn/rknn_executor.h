/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/rknn/rknn_executor.h
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-08-04:11:22:38
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_RKNN_RKNN_EXECUTOR_H__
#define __STRUCTURE_FUNCS_RKNN_RKNN_EXECUTOR_H__

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <rknn_api.h>

#include <structure/ir.h>
#include <structure/engine.h>
#include <core/utils/status.h>
#include <structure/funcs/ops.h>
#include <structure/graph_exec.h>

namespace mariana { namespace rknn {

class RknnEngine : public ::mariana::Engine {
public:
    RknnEngine();
    virtual ~RknnEngine();
    Status de_serialize(Graph& graph, const proto::ModelInfo& model_info) override;
    Status run(const ExecContext& context) override;
private:
    uint8_t* _load_data(FILE* fp, size_t ofst, size_t sz);
    uint8_t* _load_model(const std::string& filename, int* model_size);
private:
    int model_size_ = 0;
    rknn_context   ctx_;
    rknn_input_output_num io_num_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_input> rknn_inputs_;
};

}} // namespace mariana::rknn


#endif /* __STRUCTURE_FUNCS_RKNN_RKNN_EXECUTOR_H__ */

