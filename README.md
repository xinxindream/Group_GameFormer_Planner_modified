# Group_GameFormer_Planner_modified

## 一、简介
1. 组内决策规划代码，修改过模型
2. 主要记录操作流程，以及出现的各种问题
3. 在看这个文档之前，请确保[基础代码仓库](https://github.com/xinxindream/Group_GameFormer_Planner)内容已经看过，那里有前置知识

## 二、文件目录
```shell
## Planner
1. planner.py（增加了新内容）
2. planner-ori.py（原仓库文件）
```

## 三、操作流程
### 1、数据处理
使用园区数据
```shell
# 园区数据处理要用xiaoba_rosbag_tonupaln.py
python xiaoba_rosbag_tonuplan.py
--bag_path /data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part1_with_det_2.bag
--save_dir /data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part1_with_det_2_train/
```
Tips:
> 上面两个参数也可以在文件里通过设置default值来进行修改  
> 还需要去文件里修改另外四个参数：  
    &emsp;&emsp;self._original_route_lane_data_x_path  
    &emsp;&emsp;self._original_route_lane_data_y_path  
    &emsp;&emsp;self._shift_route_lane_data_x_path   
    &emsp;&emsp;self._shift_route_lane_data_y_path   
> 文件已经修改过代码，使它可以直接划分训练集和验证集  
> 可以通过修改 train_test_split(self._save_data, train_size=0.9)中的train_size设置训练集和验证集的比例  
> 结果保存路径：{save_dir}/train && {save_dir}/valid

### 2、模型训练
```shell
python train_predictor-plantf-1211.py
--name "Exp3"
--train_set /data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part1_with_det_2_train/train
--valid_set /data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part1_with_det_2_train/valid
```
Tips:
> 就是每一轮的模型都保存了，后面有评价数值，选最小的就行  
> 模型最终保存位置：./training_log/{experiment_name}/

### 3、测试
```shell
# 先启动roscore
roscore

# 执行测试文件
python xiaoba_rosbag_test.py  --model_path ./training_log/{experiment_name}/xxx.pth

# 测试进行，启动rviz
rviz -d xiaoba_rosbag_test.rviz 
```
## 四、carla仿真模拟
### 1、相关文件

### 2、操作流程
1. 数据处理&&模型训练
> 直接用[第三节]()中的结果
2. 萨芬
启动carla
cd ~ && ./start_carla.sh

启动carla_ros_bridge
cd /home/ustc/Carla/carla-ros-bridge/catkin_ws
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle_apollo_adv_ph_sr.launch

启动msg
cd /home/ustc/Carla/carla-ros-bridge/kxdun_message_ws
source ./devel/setup.bash
rosrun kxdun_planning_mpdp_pkg world_model_ros_rl_node



## 五、一些问题
1. ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found  
> https://zhuanlan.zhihu.com/p/615111375?utm_id=0  

2. carla需要低配置运行
> CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=2000 -opengl -prefernvidia -quality-level=low
> pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113  
> pkill -9 python  
> conda install -c conda-forge scikit-sparse  
> conda install -c conda-forge libstdcxx-ng=12

3. cpu_perform_command
> tuned-adm profile latency-performance  
> tuned-adm active  
> cpufreq-info  
> cat /sys/devices/system/cpu/intel_pstate/no_turbo
