注意事项：

1. 代码运行时会关闭所有终端，所以不要在终端中运行程序（可以在PyCharm中运行），其他在运行的程序也要注意先关闭；
2. 程序环境依赖在`environment.yaml`中；
3. 按照`/CarND-test/README.md`中的提示编译程序，理解`/CarND-test/src/main.cpp`的作用（可能需要作相应调整）；
4. 不同电脑，tensorflow-gpu、cuda、cudnn、keras的版本配置可能不同，请自行安装；
5. 测试仿真环境是否安装成功，可以运行`decision-making-CarND/term3_sim_linux/term3_sim.x86_64`和`decision-making-CarND/CarND-test/src/test/path_planning`；
6. 程序是利用C++与Python通过socket互传信息的方式，需要先开启Python监听，否则直接运行path_planning会显示connect error错误；
7. 样本程序所使用tensorflow、keras的版本、方式较老，如果配置有困难，可以自由选择网络搭建的方式（例如pytorch等），只要Python可以通过socket正常监听到数据即可。



环境配置步骤：

1. `git clone https://github.com/powerfulwang/decision-making-CarND.git`
2. `cd decision-making-CarND/term3_sim_linux`
3. `sudo chmod u+x term3_sim.x86_64`
4. 确保cmake >= 3.5，make >= 4.1，gcc/g++ >= 5.4（一般都已安装，未安装参考decision-making-CarND/CarND-test/README.md）
5. 进入`decision-making-CarND/CarND-test`文件夹，运行install-ubuntu.sh安装依赖（`bash install-ubuntu.sh`）
6. 保持在decision-making-CarND/CarND-test文件夹，进行编译：
   `mkdir build && cd build`
   `cmake .. && make`
   此时运行`./path_planning`显示`connect error`为正常现象
7. 安装anaconda3或miniconda3
8. 运行`conda env create -f environment.yaml`建立虚拟环境
9. 运行`decision-making-CarND/CarND-test/src/train/train.py`文件，注意事项如上

