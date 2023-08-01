/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : trt_helper.cpp
 * Authors    : wangliquan@cc-SYS-7048GR-TR
 * Create Time: 2023-07-19:18:01:10
 * Description:
 * 
 */

#include <iostream>
#include <core/utils/arrary_ref.h>
#include <structure/funcs/tensorRT/trt_helper.h>

namespace mariana { namespace trt {

void print_trt_network(const nvinfer1::INetworkDefinition& network) {
    std::cout<<"Network nodes size:" <<network.getNbLayers()<<std::endl;
    std::cout<<"Network input size:" <<network.getNbInputs()<<std::endl;
    std::cout<<"Network output size:" <<network.getNbOutputs()<<std::endl;
    for (int i = 0; i < network.getNbInputs(); ++i) {
        nvinfer1::ITensor* tensor = network.getInput(i);
        
        std::cout<<"--->input tensor:"<<i<<" name:"<<tensor->getName();
        nvinfer1::Dims dims = tensor->getDimensions();
        ArrayRef<int32_t> shape(dims.d, dims.nbDims);
        std::cout<<" shape:"<<shape<<std::endl;
    }
    for (int i = 0; i < network.getNbOutputs(); ++i) {
        nvinfer1::ITensor* tensor = network.getOutput(i);
        
        std::cout<<"--->output "<<i<<" name:"<<tensor->getName();
        nvinfer1::Dims dims = tensor->getDimensions();
        ArrayRef<int32_t> shape(dims.d, dims.nbDims);
        std::cout<<" shape:"<<shape<<std::endl;
    }
    for (int i = 0; i < network.getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network.getLayer(i);
        std::cout<<"Layer index:"<<i<<" "<<layer->getName()<<" type:"
                 <<static_cast<int>(layer->getType())<<std::endl;
        for (int n = 0; n < layer->getNbInputs(); ++n) {
            nvinfer1::ITensor* tensor = layer->getInput(n);
        
            std::cout<<"--->input "<<i<<" name:"<<tensor->getName();
            nvinfer1::Dims dims = tensor->getDimensions();
            ArrayRef<int32_t> shape(dims.d, dims.nbDims);
            std::cout<<" shape:"<<shape<<std::endl;
        }
        for (int n = 0; n < layer->getNbOutputs(); ++n) {
            nvinfer1::ITensor* tensor = layer->getOutput(n);
        
            std::cout<<"--->output "<<i<<" name:"<<tensor->getName();
            nvinfer1::Dims dims = tensor->getDimensions();
            ArrayRef<int32_t> shape(dims.d, dims.nbDims);
            std::cout<<" shape:"<<shape<<std::endl;
        }
    }
}

}} // namespace mariana::trt
