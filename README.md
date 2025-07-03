# PowerPrediction 项目说明

## 项目简介
本项目旨在使用模型通过研究用户过去的用电情况来预测未来的用户电力消耗情况。使用的模型包括：

**LSTM 模型**

**Transformer 模型**


**自定义模型：Autoformer**

支持两种预测任务：

**短期预测 (horizon=90)**：基于过去90天的数据预测未来90天的的用户电力消耗情况。

**长期预测 (horizon=365)**：基于过去90天的数据预测未来365天的用户电力消耗情况。



***

## 数据说明

项目使用的初始训练数据和测试数据存放于`data/`文件夹内：

`train.csv`

`test.csv`

### 数据处理过程
通过运行`python data/data_pre.py`来查看训练集和测试集中的数据属性，以及每列中空值情况。

通过运行`python data/process.py`将原来的训练集和测试集变为以天为单位并对数据集中的剩余属性分别求和或者取均值或者取第一个值。并通过公式计算剩余电表的电力消耗情况。


***

## 环境依赖

运行本项目需要通过下面的命令来安装环境依赖。

`pip install -r requirements.txt`


***

## 快速开始

通过以下命令训练不同模型，并完成预测任务：

### 1. LSTM 模型

**短期预测 (horizon=90)**

```
python main.py --model LSTM --horizon 90
```

**长期预测 (horizon=365)**



```
python main.py --model LSTM --horizon 365
```

### 2. Transformer 模型

**短期预测 (horizon=90)**



```
python main.py --model Transformer --horizon 90
```

**长期预测 (horizon=365)**



```
python main.py --model Transformer --horizon 365
```



### 3. 自定义模型：Autoformer

**短期预测 (horizon=90)**



```
python main.py --model Autoformer --horizon 90
```

**长期预测 (horizon=365)**



```
python main.py --model Autoformer --horizon 365
```

### 运行过程记录


每次运行都会生成自动创建`model`、`result`、`predict`、`output`四个文件夹，其中`model`文件夹中保存每次训练的模型和使用的缩放模型、`result`
文件夹保存生成的预测和真实值之间的曲线对比图、`predict`文件夹保存的是模型预测的具体结果，方便后续比较三个模型的预测情况、`output`文件夹中保存的是每次运行的结果.json文档，其中保存了运行的模型、预测天数、平均MSE和平均MAE。文件夹中个文件的命名方式包括模型名称和预测天数，保证文件名的唯一性。


### 各个绘图py文件

直接运行即可

### 示例


```
python plot_results.py 
```

