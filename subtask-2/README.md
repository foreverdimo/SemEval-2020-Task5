<font size=10>**SemEval-2020 Task5-subtask2**</font>



算法设计与分析 2020  期末项目



## 主要库的版本

本项目是基于keras（Using TensorFlow backend）以下是主要库的版本

- python = 3.7.4（Anaconda3)

- keras == 2.2.4
- keras_contrib == 0.0.2
- keras_bert == 0.81.0
- tensorflow == 1.14.0



## 项目说明

本项目将序列标注问题转换为实体识别问题，将句子中词语分为5类，分别是“Normal”，“Antecedent Start"、"Antecedent End”、"Consequence Start"、"Consequence End"，利用 Bert、双向LSTM和CRF模型进行训练。

分别使用了 Google 预训练的Bert 模型 和  没有预训练的 Bert 模型 进行训练。

没有预训练的版本在项目文件夹里以“Non-pretrained”子目录存在



- data 数据目录
  - data：训练数据
  - uncased_L-12_H-768_A-12： Google Bert的预训练模型
  - vocab：单词字典
- log 日志目录
  - train_log : 记录训练详细过程
  - df.csv : 训练结果
- pics : 有关训练的图示
- Bert+BiLSTM+CRF.ipynb/html ： Jupyter Notebook 版展示训练结果
- model/Non-pretrained-model.py :  项目源代码

## 运行项目

安装项目所需的库后，命令行运行 train.py 或者 jupyter notebook 运行 ipynb文件即可

## 运行结果和分析

Pre- trained- model:   GPU： Nvidia GeForceRTX2070 with Max-Q Design         

Non-pre-trained-model : GPU : Nvidia Tesla V100  (from Baidu Aistudio)



训练周期为15个周期，提前停止条件：2个周期验证集准确率没有提升。

batch_size=32



训练结果： 在验证集上的准确率为 0.988

​					F1 score = 0.198 ， Recall  = 0.197

(也可在 log 文件里 查看)



 分析： 在验证集上的准确率已经比较高了，但是F1分比较低，因为Recall 分比较低，可能是因为直接将模型转化			  为实体识别模型直接计算会出现正例偏低的情况（因为绝大多数词都是Normal类），如果再转换回序列                       			  标注可能得分会升高。其次 Bert更注重于句子的结构和单词之间的联系，而对于条件的判断很重要的词		   （例如 could，not，have)等一类词并没有给出很高的attention ，但总体而言表现还是相对优秀。另外训 			  练集也相对较小。鉴于时间原因，还没有进行相关改进工作。

