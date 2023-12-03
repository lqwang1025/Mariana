/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_softmax_io_reshape.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-10-17:18:00:02
 * Description:
 * 
 */

#include <maro/transform_utils.h>
#include <structure/ir.h>
#include <core/utils/logging.h>
#include <structure/funcs/ops.h>
#include <structure/funcs/softmax.h>
#include <structure/funcs/reshape.h>
#include <structure/funcs/register.h>

namespace mariana { namespace transform {

/*
 *        *    <==== reshape if input dim == 4
 *        |  
 *      SfotMax
 *        |
 *        *    <==== reshape
 */

Status trt_softmax_io_reshape(Graph& graph) {
    auto func = [](const NodeMatch& match,
                   std::set<std::string>* old_nodes,
                   std::vector<std::shared_ptr<Node>>* new_nodes) -> Status {
        MVLOG(2)<<"Match:"<<match.debug_string();
        std::shared_ptr<Node> softmax_node = match.node;
        std::shared_ptr<Node> inode = inodes_of(softmax_node)[0];

        Shape ishape = inode->shapes()[softmax_node->ctrl_idx()[0]];
        if (ishape.dims() == 4) {
            register_funcs();

            SoftmaxFunction* softmax_func = static_cast<SoftmaxFunction*>(softmax_node->op());
            uint32_t axis = softmax_func->option.axis;
            if (axis == 3) {
                std::string reshape_iname = softmax_node->name()+"_reshape_i";
                std::string reshape_oname = softmax_node->name()+"_reshape_o";
                std::shared_ptr<Node> reshape_inode = std::make_shared<Node>(*softmax_node->graph());
                std::shared_ptr<Node> reshape_onode = std::make_shared<Node>(*softmax_node->graph());
                reshape_inode->init(reshape_iname, MRESHAPE);
                reshape_onode->init(reshape_oname, MRESHAPE);
                ReshapeFunction* re_ifunc = static_cast<ReshapeFunction*>(reshape_inode->op());
                re_ifunc->option.shape.resize(2);
                re_ifunc->option.shape[1] = ishape[3];
                int64_t pro = 1;
                for (int i = 0; i < ishape.dims()-1; ++i) {
                    pro *= ishape[i];
                }
                re_ifunc->option.shape[0] = pro;
                reshape_inode->shapes().push_back(Shape(re_ifunc->option.shape));
                reshape_inode->ctrl_idx()    = softmax_node->ctrl_idx();
                softmax_node->ctrl_idx().push_back(0);
                reshape_inode->inputs().push_back(inode->name());
                
                softmax_node->inputs()[0] = reshape_inode->name();
                softmax_func->option.axis = 1;
                
                ReshapeFunction* re_ofunc = static_cast<ReshapeFunction*>(reshape_onode->op());
                re_ofunc->option.shape.reserve(ishape.dims());
                for (int i = 0; i < ishape.dims(); ++i) {
                    re_ofunc->option.shape.push_back(ishape[i]);
                }
                reshape_onode->shapes().push_back(Shape(re_ofunc->option.shape));
                reshape_onode->ctrl_idx().push_back(0);
                reshape_onode->inputs().push_back(softmax_node->name());
                std::vector<std::shared_ptr<Node>> onodes = onodes_of(softmax_node);
                for (auto& it : onodes) {
                    for (size_t i = 0; i < it->inputs().size(); ++i) {
                        if (it->inputs()[i] == softmax_node->name()) {
                            it->inputs()[i] = reshape_onode->name();
                        }
                    }
                }
                new_nodes->push_back(reshape_inode);
                new_nodes->push_back(reshape_onode);
            }
        }
        unregister_funcs();
        return absl::OkStatus();
    };
    replace_matching_optypes(graph,
                             {"SOFTMAX",
                                 {}
                             }, func);
    return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("trt_softmax_io_reshape", trt_softmax_io_reshape);

}} // namespace mariana::transform
