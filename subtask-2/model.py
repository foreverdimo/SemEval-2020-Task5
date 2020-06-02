## Group project for DAA fianl 
## By Yiyang Zhang & Yumin Jiang

import keras
from keras.models import Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping
import keras_bert
import logging
from sklearn.metrics import f1_score, recall_score
import numpy as np
import pandas as pd
import os
import re 
from pandas.core.frame import DataFrame



# Get file path

# Get current file_path
current_dir = os.path.dirname(os.path.abspath('__file__'))
# Vocab path
path_vocab = os.path.join(current_dir, 'data/vocab/vocab.txt')

# Source- Data path

source_data_dir = os.path.join(current_dir, 'data/data/train.xlsx')

# Data path
path_data_dir = os.path.join(current_dir, 'data/data/')

# Bert pre-trained model path
path_bert_dir = os.path.join(current_dir, 'data/uncased_L-12_H-768_A-12/')

#log path
path_log_dir = os.path.join(current_dir, "log/")


# Global Variables and Parameters Settings

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'

max_len = 300
batch_size = 16
train_epoch = 15 
drop_rate = 0.15
valid_rate = 0.2

# Data pre-processing
# For convenience , We have converted train.csv to train.xlsx
# Make source-data more convenient to process for Bert model

tian_pd = pd.read_excel(source_data_dir)

dc_list = []
zt_q_list = []
for i in range(len(tian_pd)):
    a = tian_pd["b"][i].lower()
    b = tian_pd["c"][i].lower()
    c = tian_pd["d"][i].lower()
    
    #e_list = a.split(" ")
    e_2_list = []
    e_list =a.split(" ")
    for i in e_list:
        if i == "don't":
            e_2_list += ["do","not"]
        else:
            a_list = re.findall(r"[\w']+|[.,!?;-]|[\\#\\@\\:\"\(\)]",i)
            e_2_list += a_list
        e_2_list.append("[unused0]")
            
    
        
    b1 = re.compile(r"\b[a-zA-Z]+\b",re.I).findall(b)
    c1 = re.compile(r"\b[a-zA-Z]+\b",re.I).findall(c)
    AS = b1[0]
    AE = b1[-1]
    CS = c1[0] if c1 != [] else ""
    CE = c1[-1] if c1 != [] else ""
    zt_list = []
    same = []
    for o,p in enumerate(e_2_list):
        if p not in same and AS == p:
            same.append(p)
            zt_list.append("AS")
        elif p not in same and AE == p:
            same.append(p)
            zt_list.append("AE")
        elif p not in same and CS == p:
            same.append(p)
            zt_list.append("CS")
        elif p not in same and CE == p:
            same.append(p)
            zt_list.append("CE")
        else:
            zt_list.append("N")
    
    
    e_2_list.append(" ")
    zt_list.append(" ")
    dc_list += e_2_list
    zt_q_list += zt_list

f={"a" : dc_list,
   "b" : zt_q_list}#
data=DataFrame(f)#
data.to_csv("data/data/train.csv",header=False,index=False)


data = pd.read_csv('data/data/train.csv', encoding='utf-8',header = None)
total_num = 3551
train_num = (int)(total_num * (1-valid_rate))
valid_num = total_num - train_num
count = 0
temp_index = 0

with open('data/data/train-2.txt','w+', encoding='utf-8') as f:
    for i in range(len(data.values)):
        if str(data.values[i][0]) == ' ' :
            f.write('\n')
            count += 1
            if count == train_num:
                temp_index = i
                break
        else :
            f.write((str(data.values[i][0])+' '+str(data.values[i][1])+'\n'))
            
with open('data/data/valid-2.txt','w+', encoding='utf-8') as f:
    for i in range(temp_index + 1, len(data.values)):
        if str(data.values[i][0]) == ' ' :
            f.write('\n')
        else :
            f.write((str(data.values[i][0])+' '+str(data.values[i][1])+'\n'))
print("Data pre-processing Complete")            
print("Train on "+str(train_num)+" examples")
print("Validate on "+str(valid_num)+" examples")

# Helper Functions

# Get Word to Index Dictionary
def get_w2i(vocab_path = path_vocab):
    w2i = {}
    with open(vocab_path, 'r',encoding='UTF-8') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


# Get Tag to Index Dictionary
def get_tag2index():
    return {"N": 0,
            "AS": 1, "AE": 2,
            "CS": 3, "CE": 4,
            }

# Process Data

class DataProcess(object):
    def __init__(self,
                 max_len=100,
                 data_type='data',
                 model='bert',
                 ):
        
        self.w2i = get_w2i()  # word to index
        self.tag2index = get_tag2index()  # tag to index
        
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.max_len = max_len
        
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)
        
        
        self.model = model
        self.base_dir = path_data_dir

    def get_data(self, one_hot: bool = True) -> ([], [], [], []):

        # Concatenate File_path
        path_train = os.path.join(self.base_dir, "train-2.txt")
        path_test = os.path.join(self.base_dir, "valid-2.txt")

        train_data, train_label = self.__bert_text_to_index(path_train)
        test_data, test_label = self.__bert_text_to_index(path_test)

        # One-hot Processing
        if one_hot:
            def label_to_one_hot(index: []) -> []:
                data = []
                for line in index:
                    data_line = []
                    for i, index in enumerate(line):
                        line_line = [0]*self.tag_size
                        line_line[index] = 1
                        data_line.append(line_line)
                    data.append(data_line)
                return np.array(data)
            train_label = label_to_one_hot(index=train_label)
            test_label = label_to_one_hot(index=test_label)
        else:
            train_label = np.expand_dims(train_label, 2)
            test_label = np.expand_dims(test_label, 2)
        return train_data, train_label, test_data, test_label

    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    def __bert_text_to_index(self, file_path: str):
        """
        bert的数据处理
        处理流程 所有句子开始添加 [CLS] 结束添加 [SEP]
        bert需要输入 ids和types所以需要两个同时输出
        由于我们句子都是单句的，所以所有types都填充0

        :param file_path:  文件路径
        :return: [ids, types], label_ids
        """
        data_ids = []
        data_types = []
        label_ids = []
        with open(file_path, 'r',encoding='UTF-8') as f:
            line_data_ids = []
            line_data_types = []
            line_label = []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    # bert 需要输入index和types 由于是单语句模型，所以type都为0
                    w_index = self.w2i.get(w, self.unk_index)
                    t_index = self.tag2index.get(t, 0)
                    line_data_ids.append(w_index)  # index
                    line_data_types.append(0)  # types
                    line_label.append(t_index)  # label index
                else:
                    # 处理填充开始和结尾 bert 输入语句每个开始需要填充[CLS] 结束[SEP]
                    max_len_buff = self.max_len-2
                    if len(line_data_ids) > max_len_buff: # 先进行截断
                        line_data_ids = line_data_ids[:max_len_buff]
                        line_data_types = line_data_types[:max_len_buff]
                        line_label = line_label[:max_len_buff]
                    line_data_ids = [self.cls_index] + line_data_ids + [self.sep_index]
                    line_data_types = [0] + line_data_types + [0]
                    line_label = [0] + line_label + [0]

                    # padding
                    if len(line_data_ids) < self.max_len: # 填充到最大长度
                        pad_num = self.max_len - len(line_data_ids)
                        line_data_ids = [self.pad_index]*pad_num + line_data_ids
                        line_data_types = [0] * pad_num + line_data_types
                        line_label = [0] * pad_num + line_label
                    data_ids.append(np.array(line_data_ids))
                    data_types.append(np.array(line_data_types))
                    label_ids.append(np.array(line_label))
                    line_data_ids = []
                    line_data_types = []
                    line_label = []
        return [np.array(data_ids), np.array(data_types)], np.array(label_ids)


# Bert-BiLSTM-CRF model

class BERTBILSTMCRF(object):
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 max_len: int = 100,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate
        self.config_path = os.path.join(path_bert_dir, 'bert_config.json')
        self.check_point_path = os.path.join(path_bert_dir, 'bert_model.ckpt')
        self.dict_path = os.path.join(path_bert_dir, 'vocab.txt')

    def creat_model(self):
        print('load bert Model start!')
        model = keras_bert.load_trained_model_from_checkpoint(self.config_path,
                                                              checkpoint_file=self.check_point_path,
                                                              seq_len=self.max_len,
                                                              trainable=False)
        print('load bert Model end!')
        inputs = model.inputs
        embedding = model.output
        print("Inputs shape:"+str((np.array(inputs)).shape))
        print("Embedding shape:"+str(embedding.shape))
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(embedding)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.n_class)(x)
        self.crf = CRF(self.n_class, sparse_target=False)
        x = self.crf(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(optimizer=Adam(1e-5),
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])

# Log Settings

def create_log(path, stream=True):
    """
    获取日志对象
    :param path: 日志文件路径
    :param stream: 是否输出控制台
                False: 不输出到控制台
                True: 输出控制台，默认为输出到控制台
    :return:日志对象
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)
    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


class TrainHistory(keras.callbacks.Callback):
    def __init__(self, log=None, model_name=None):
        super(TrainHistory, self).__init__()
        if not log:
            path = os.path.join(path_log_dir, 'callback.log')
            log = create_log(path=path, stream=False)
        self.log = log
        self.model_name = model_name
        self.epoch = 0
        self.info = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        message = f"begin epoch: {self.epoch}"
        self.log.info(message)

    def on_epoch_end(self, epoch, logs={}):
        message = f'end epoch: {epoch} loss:{logs["loss"]} val_loss:{logs["val_loss"]} acc:{logs["crf_viterbi_accuracy"]} val_acc:{logs["val_crf_viterbi_accuracy"]}'
        self.log.info(message)
        dict = {
            'model_name':self.model_name,
            'epoch': self.epoch+1,
            'loss': logs["loss"],
            'acc': logs['crf_viterbi_accuracy'],
            'val_loss': logs["val_loss"],
            'val_acc': logs['val_crf_viterbi_accuracy']
        }
        self.info.append(dict)

    def on_batch_end(self, batch, logs={}):
        message = f'{self.model_name} epoch: {self.epoch} batch:{batch} loss:{logs["loss"]}  acc:{logs["crf_viterbi_accuracy"]}'
        self.log.info(message)


# Train Process
def train_sample(train_model='BERTBILSTMCRF',
                 epochs= 10,
                 log = None
                 ):

    dp = DataProcess(data_type='msra', max_len=max_len, model='bert')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    model = BERTBILSTMCRF(dp.vocab_size, dp.tag_size, max_len=max_len, drop_rate = drop_rate)

    model = model.creat_model()

    callback = TrainHistory(log=log, model_name=train_model)  
    early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max') 
    model.fit(train_data, train_label, batch_size= batch_size, epochs=epochs,
              validation_data=[test_data, test_label],
              callbacks=[callback, early_stopping])

    # compute f1 & recall val

    pre = model.predict(test_data)
    pre = np.array(pre)
    test_label = np.array(test_label)
    pre = np.argmax(pre, axis=2)
    test_label = np.argmax(test_label, axis=2)
    pre = pre.reshape(pre.shape[0] * pre.shape[1], )
    test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], )

    f1score = f1_score(pre, test_label, average='macro')
    recall = recall_score(pre, test_label, average='macro')

    log.info("================================================")
    log.info(f"--------------:f1: {f1score} --------------")
    log.info(f"--------------:recall: {recall} --------------")
    log.info("================================================")

    info_list = callback.info
    if info_list and len(info_list)>0:
        last_info = info_list[-1]
        last_info['f1'] = f1score
        last_info['recall'] = recall

    return info_list



train_modes = ['BERTBILSTMCRF']

log_path = os.path.join(path_log_dir, 'train_log.log')
df_path = os.path.join(path_log_dir, 'df.csv')
log = create_log(log_path)
columns = ['model_name','epoch', 'loss', 'acc', 'val_loss', 'val_acc', 'f1', 'recall']
df = pd.DataFrame(columns=columns)
for model in train_modes:
    info_list = train_sample(train_model=model, epochs=train_epoch, log=log)
    for info in info_list:
        df = df.append([info])
df.to_csv(df_path)
