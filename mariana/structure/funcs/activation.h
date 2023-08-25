/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/funcs/activation.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-29:11:09:43
 * Description:
 *
 */

#ifndef __STRUCTURE_FUNCS_ACTIVATION_H__
#define __STRUCTURE_FUNCS_ACTIVATION_H__

#include <vector>
#include <cstdint>

#include <structure/function.h>
#include <structure/func_option.h>

namespace mariana {

enum class ActivationType : int8_t {
    UNINIT = -1,
    kRELU = 0,             //!< Rectified linear activation.
    kSIGMOID = 1,          //!< Sigmoid activation.
    kTANH = 2,             //!< TanH activation.
    kLEAKY_RELU = 3,       //!< LeakyRelu activation: x>=0 ? x : alpha * x.
    kELU = 4,              //!< Elu activation: x>=0 ? x : alpha * (exp(x) - 1).
    kSELU = 5,             //!< Selu activation: x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
    kSOFTSIGN = 6,         //!< Softsign activation: x / (1+|x|)
    kSOFTPLUS = 7,         //!< Parametric softplus activation: alpha*log(exp(beta*x)+1)
    kCLIP = 8,             //!< Clip activation: max(alpha, min(beta, x))
    kHARD_SIGMOID = 9,     //!< Hard sigmoid activation: max(0, min(1, alpha*x+beta))
    kSCALED_TANH = 10,     //!< Scaled tanh activation: alpha*tanh(beta*x)
    kTHRESHOLDED_RELU = 11 //!< Thresholded ReLU activation: x>alpha ? x : 0
};
    
struct ActivationOption : public BaseOption {
    ActivationOption() {}
    ~ActivationOption() {}
    ActivationType act_type = ActivationType::UNINIT;
};

struct ActivationFunction : public Function {
    ActivationFunction() {}
    ~ActivationFunction() {}
    ActivationOption option;
    tensor_list compute(tensor_list&& inputs) override;
    ShapeList infer_shape(ShapeList shapes) override;
};

} // namespace mariana

#endif /* __STRUCTURE_FUNCS_ACTIVATION_H__ */

