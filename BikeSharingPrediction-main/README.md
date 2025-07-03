# BikeSharingPrediction 项目说明

## 项目简介

本项目旨在基于历史共享单车租赁数据，预测未来的租赁数量。使用的模型包括：

**LSTM 模型**

**Transformer 模型**

**自定义模型：TFT**

**自定义模型：Autoformer**

支持两种预测任务：

**短期预测 (O=96)**：基于过去 96 小时的数据预测未来 96 小时的租赁数量。

**长期预测 (O=240)**：基于过去 96 小时的数据预测未来 240 小时的租赁数量。



***

## 数据说明

项目使用的训练数据和测试数据存放于`data/`文件夹内：

`train_data.csv`

`test_data.csv`



***

## 环境依赖

运行本项目需要以下环境依赖：

Python 3.8+

PyTorch 1.13+

其他依赖自行安装。



***

## 快速开始

通过以下命令训练不同模型，并完成预测任务：

### 1. LSTM 模型

**短期预测 (O=96)**



```
python main.py --model LSTM --output_window 96
```

**长期预测 (O=240)**



```
python main.py --model LSTM --output_window 240
```

### 2. Transformer 模型

**短期预测 (O=96)**



```
python main.py --model Transformer --output_window 96
```

**长期预测 (O=240)**



```
python main.py --model Transformer --output_window 240
```

### 3. 自定义模型：TFT

**短期预测 (O=96)**



```
python main.py --model TFT --output_window 96
```

**长期预测 (O=240)**



```
python main.py --model TFT --output_window 240
```

### 4. 自定义模型：Autoformer

**短期预测 (O=96)**



```
python main.py --model Autoformer --output_window 96
```

**长期预测 (O=240)**



```
python main.py --model Autoformer --output_window 240
```

### 日志记录

每次运行会自动生成日志文件，保存于项目根目录。日志文件命名格式为：



```
training_<model_name>_output<output_window>.log
```

日志文件中包含每次运行的训练损失和测试结果（MSE、MAE）。

### 各个绘图py文件

直接运行即可

### 示例


```
python plot_results.py 
```

