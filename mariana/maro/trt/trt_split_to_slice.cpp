/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : split_to_slice.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-11:13:14:40
 * Description:
 * 
 */

#include <maro/transform_utils.h>
#include <structure/ir.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/funcs/slice.h>
#include <structure/funcs/split.h>
#include <structure/funcs/register.h>

namespace mariana { namespace transform {

/*
 *    ReshapeInput     AnyFusedReshape
 *     /  |  \            / | \
 *      Reshape  =======>   |
 *        |                 |
 *      AnyNode          AnyNode
 */

Status trt_split_to_slice(Graph& graph) {
    auto func = [](const NodeMatch& match,
                   std::set<std::string>* old_nodes,
                   std::vector<std::shared_ptr<Node>>* new_nodes) -> Status {
        MVLOG(2)<<"Match:"<<match.debug_string();
        std::shared_ptr<Node> split_node = match.node;
        std::shared_ptr<Node> split_input_node = match.inputs[0].node;
        

        std::vector<int32_t> splits = {0};
        SplitFunction* split_func = static_cast<SplitFunction*>(split_node->op());
        int32_t tmp_num = 0;
        for (auto& it : split_func->option.split) {
            tmp_num += it;
            splits.push_back(tmp_num);
        }
        register_funcs();


        /*
         *   Split
         *    / \
         *   CONCAT
         *    
         */
        std::vector<std::shared_ptr<Node>> onodes = onodes_of(split_node);
        std::set<std::shared_ptr<Node>> s(onodes.begin(), onodes.end());
        onodes.assign(s.begin(), s.end());
        for (size_t i = 0; i < onodes.size(); ++i) {
            for (size_t j = 0; j < onodes[i]->inputs().size(); ++j) {
                if (onodes[i]->inputs()[j] == split_node->name()) {
                    std::string slice_name = split_node->name()+"_"+std::to_string(i)+"_"+std::to_string(j);
                    std::shared_ptr<Node> slice_node = std::make_shared<Node>(*split_node->graph());
                    slice_node->init(slice_name, MSLICE);
                    SliceFunction* slice_func = static_cast<SliceFunction*>(slice_node->op());
                    int32_t        ctrl_index = onodes[i]->ctrl_idx()[j];
                    slice_node->shapes().push_back(split_node->shapes()[ctrl_index]);
                    slice_func->option.begin  = splits[ctrl_index];
                    slice_func->option.end    = splits[ctrl_index+1];
                    slice_func->option.axis   = split_func->option.axis;
                    slice_node->inputs()      = split_node->inputs();
                    slice_node->ctrl_idx()    = split_node->ctrl_idx();
                    onodes[i]->inputs()[j]    = slice_name;
                    onodes[i]->ctrl_idx()[j]  = 0;
                    new_nodes->push_back(slice_node);
                }
            }
        }
        old_nodes->insert(split_node->name());
        unregister_funcs();
        return absl::OkStatus();
    };
    replace_matching_optypes(graph,
                             {"SPLIT",
                                 {
                                     {"*", {}}
                                 }
                             }, func);
    return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("trt_split_to_slice", trt_split_to_slice);

}} // namespace mariana::transform
