该项目为使用Mariana库在rknn或tensorRT上进行yolov8+deepsort部署的示例
在rknn上部署yolov8+deepsort的步骤如下：
1、【修改yolov8导出模型】：进入文件ultralytics/nn/modules 修改line:61 代码：y = torch.cat((dbox, cls.sigmoid()), 1) 为  y = x_cat.transpose(1,2)
2、下载yolov8预训练模型，这里以yolov8s为例，执行yolo export model=./yolov8s.pt  format=onnx simplify batch=1 opset=12
3、【预先安装rknn-toolkit 最低版本为1.7.3】shell 指定 python convert.py
4、【该步骤执行前，需将librknn_api库软链接到3rd_party(没有3rd_party这个目录需要手动创建)】编译安装Mariana库，进入mariana目录执行：mkdir build && cd build && cmake -D CMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake  -D WITH_RKNN=ON ..
5、进入mariana/sample/deepsort 目录：mkdir build && cd build && cmake.. && make install
6、enjoy it.
