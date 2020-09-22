## Train_Crlc
##### TRAIN_CRLC用于训练网络
##### cnn_CRLC_v2用于测试网络模型和AOM工程调用
##### UTILS是一些加载数据，处理数据相关的方法
##### 其他代码是一些网络结构

##### 此代码用于训练多通道输出网络，网络的输出会进行线性组合，即CRLC_v**.crlc_model
	
#### 训练方法：
	使用TRAIN_CRLC.py训练
	1. 更改 from CRLC_v** import crlc_model as model 中的**选择不同的模型
	2. 设置训练集的路径，LOW_DATA_PATH的路径中包括多个QP的数据，而HIGH_DATA_PATH只有一份
	3. 在UTILS.py中的get_train_list中修改 if语句的范围 ，选择要训练的QP范围
	4. 设置学习率，迭代次数等参数，开始训练
		
#### 环外测试方法：
	crlc： 使用cnn_CRLC_v2.py
	1.更改main函数中test_all_ckpt(r“***”)，***中的路径
	2.更改 from model.CRLC_v** import crlc_model as model 中的**选择不同的模型
	3.设置 low_img 路径
	4.测试选择最好模型
