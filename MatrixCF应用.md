## MatrixCF在召回和排序中的运用

### 1. 工业推荐架构

![推荐系统架构](https://github.com/RTCFoundation/RecSys/blob/main/images/推荐系统架构.jpg)

特征服务：存放用户特征和物料特征的一个存储容器

向量服务：存放用户向量和物料向量的一个存储容器

参数服务：存储模型训练后的参数信息

频控服务：对召回候选集进行一些过滤，如把用户曝光/点击过的文章过滤掉

机制服务：类别打散，人为干预，展示逻辑

#### 召回服务

![RecallService](https://github.com/RTCFoundation/RecSys/blob/main/images/RecallService.jpg)

Recall Service对接引擎请求，拿到引擎的请求信息，基于这些信息，并行获取各路召回分支的召回结果

获取的方式

按功能划分为：

从召回候选池（通常是从redis）中抽取召回结果（外部召回集）

直接内部进行一些在线实时计算获得召回结果（内部召回集）

按算法划分为：

基于用户兴趣标签召回

协同过滤召回

向量召回

深度匹配召回

#### 排序服务

排序服务与在线预估服务不一样

在线预估服务实际上是调用排序服务模型训练的结果模型和参数，进行在线预估分值

排序服务是一个提供各种模型训练的平台，如排序模型FM（Factorization Machine），在平台上训练后统一部署到线上以便给在线预估服务使用

![SortingService](https://github.com/RTCFoundation/RecSys/blob/main/images/SortingService.jpg)

1）在线排序

示例：基于FM模型在线排序服务

- 开发FM模型

- 收集kafka的训练样本达到一个batch，触发训练

- 从参数服务中拉取下来与该batch相关的参数，然后基于准备的样本进行训练

- 更新参数后，并上传到参数服务上

- 重复上面的两步，形成一个在线排序服务，该过程回随着服务的部署一直进行训练

![onlinesort](https://github.com/RTCFoundation/RecSys/blob/main/images/onlinesort.jpg)

2) 离线排序

示例：基于FM模型在线排序服务

- 开发FM模型

- 准备训练样本数据sample，该样本数据会在离线处理成大量的batch数据单元

- 基于这些训练样本数据进行模型训练，然后保存模型

- 上传模型到在线预估服务平台

![offlinesort](https://github.com/RTCFoundation/RecSys/blob/main/images/offlinesort.jpg)

跟在线排序不同点：训练样本的准备，离线是一次准备好所有数据

在线和离线的区别

1. 训练样本的区别

   离线训练是一次基于所有的历史数据构建所有训练样本，而在线训练随着时间不断的进行构建样本，是一个递增的过程

2. 模型训练的区别

   离线训练基于所有训练样本迭代训练一定次数就会结束，而在线训练不会结束，会一直基于新增样本进行训练

3. 最终结果保存的区别

   离线训练通常是保存一个模型文件或参数文件，然后部署到在线预估服务上

   在线训练通常不断更新参数服务的参数从而实现对模型的更新，在线服务直接调用参数服务进行预估

### 2. MatrixCF运用于召回

#### 2.1 获取原始数据

这里直接将ratings.dat数据放在了data目录下

#### 2.2 原始数据的预处理

原始数据格式为UserID::MovieID::Score::TS

处理方式如下

- 抽取用户行为数小于2000的用户

- 基于时间戳排序后，划分训练样本和测试样本

  取后2个行为作为测试样本，剩余的训练样本

- 对数据只取UserID，movieID，score

备注：用户行为数表示一个用户对多少个电影进行评分

```python
data_file = "./data/ratings.dat"
#加载文件
lines = open(data_file)

#统计数据量
count = 0
for line in lines:
	count += 1
print("数据总量: ", count)         
```

```python
# step1: 读取数据，数据格式为{uid: [{mid, score, ts}]}
def read_raw_data(file_path):
  	user_info = dict()
    lines = open(file_path)
    for line in lines:
      	tmp = line.strip().split("::")
        if len(tmp) < 4:
            continue
        uid = user_info.get(tmp[0], None)
        if uid is None:
          #之前用户不存在
          user_info[tmp[0]] = [(tmp[1], tmp[2], tmp[3])]
         else:
          #用户存在则追加
          user_info[tmp[0]].append((tmp[1], tmp[2], tmp[3]))
          
     return user_info
```

```python
# step2:抽取用户行为数小于2000的用户
def extract_valid_user(user_info):
  user_info_filter = {}
  for k, v in user_info.items():
    if (len(v) > 2000):
      continue
    user_info_filter[k] = v
  return user_info_filter

user_info = extract_valid_user(user_info)
```

```python
# step3:基于时间戳对行为序列进行排序，并划分训练集和测试集
def split_train_test(user_info):
  train_set = []
  test_set = []
  for k, v in user_info.items():
    tmp = sorted(v, key=lambda _: _[2]) #基于时间排序
    for i in range(len(tmp)):
      if i < len(tmp) - 2:
        train_set.append(str(k) + "," + tmp[i][0] + "," + tmp[i][1])
      else:
        test_set.append(str(k) + "," + tmp[i][0] + "," + tmp[i][1])
  return train_set, test_set

train_set, test_set = split_train_test(user_info)
```

```python
# step4: 保存训练数据和验证数据（测试数据）
def save_data(train_set, test_set, save_path_dir):
  import random
  random.shuffle(train_set)
  random.shuffle(test_set)
  with open(save_path_dir + "train_set", "w") as f:
    for line in train_set:
      f.write(line + "\n")
   
  with open(save_path_dir + "test_set", "w") as f:
    for line in test_set:
      f.write(line + "\n")
 
save_path = "./data/"
save_data(train_set, test_set, save_path)
```

#### 2.3 转换数据格式，适用于模型训练

##### 2.3.1 数据hash化

hash离散化的作用

a.更好的稀疏表示

b.具有一定的特征压缩

c.能够方便模型的部署

处理过程：

- 对数据中特征进行hash化处理

- 特征hash化的逻辑

  hash_fun(特征域=特征值)如Hash("user_id=1")

```python
# 定义hash函数
def bkdr2hash64(str):
	mask = 0x0fffffffffffffff
  seed = 131
  hash = 0
  for s in str:
    hash = hash * seed + ord(s)
  return hash & mask
```

```python
def tohash(file, save_path):
  wfile = open(save_path, "w")
  with open(file) as f:
    tmp = line.strip().split(",")
    user_id = bkdr2hash64("UserID=", tmp[0])
    item_id = bkdr2hash64("MovieID=", tmp[1])
    wfile.write(str(user_id) + "," + str(item_id) + "," + tmp[2])
  wfile.close()
  
train_file_path = "./data/train_set"
train_tohash = "./data/train_set_tohash"
tohash(train_file_path, train_tohash)

test_file_path = "./data/test_set"
test_tohash = "./data/test_set_tohash"
tohash(test_file_path, test_tohash)
```

##### 2.3.2 tfrecords

tfrecords是一种序列化数据格式，适用于tensorflow框架中的模型训练使用

转成tfrecords数据格式如下

```python
Example {
	"feature" : tf.int64_list
	"label" : tf.float_list
}
```

```python
import tensorflow as tf
# 定义tfrecords格式
def get_tfrecords_example(feature, label):
  tfrecords_features = {
    'feature': tf.train.Feature(int64_list = tf.train.Int64List(value = feature)),
    'label': tf.train.Feature(float_list = tf.train.FloatList(value = label))
  }
  
  return tf.train.Example(
  				features = tf.train.Features(feature = tfrecords_features))
```

```python
# 对训练样本和测试样本进行转换
def totfrecords(file, save_dir):
  print("Process to tfrecord Files: %s ..." %file)
  num = 0
  writer = tf.io.TFRecordWriter(save_dir + "/" + "part-0000" + str(num) + ".tfrecords")
  lines = open(file)
  for i, line in enumerate(lines):
    tmp = line.strip().split(",")
    feature = [int(tmp[0]), int(tmp[1])]
    #label是二值label，只有0，1
    label = [float(1) if float(tmp[2]) >= 3 else float(0)]
    example = get_tfrecods_example(feature, label)
    writer.write(example.SerializeToString())
    if (i + 1) % 200000 == 0:
      writer.close()
      num += 1
      # 模拟离线hive数据，spark情况
      writer = tf.io.TFRecordWriter(save_dir + "/" + "part-0000" + str(num) + ".tfrecords")
    print("Process to tfrecord File: %s End" % file)
    writer.close()
    
# 具体调用方法
import os

train_file_path = "./data/train_set_tohash"
train_totfrecord = "./data/train"
test_file_path = "./data/test_set_tohash"
test_totfrecord = "./data/val"

os.mkdir(train_totfrecord)
os.mkdir(test_totfrecord)
totfrecords(train_file_path, train_totfrecord)
totfrecords(test_file_path, test_totfrecord)
```

#### 2.4 模型开发及训练

#### ![模型架构图](https://github.com/RTCFoundation/RecSys/blob/main/images/模型架构图.jpg)

##### 2.4.1 Parameter Server

这个是一个简单的PS，使用了一个字典存储

注意点：不是一开始对所有的特征随机向量，而是用到的时候才随机产生的

```python
import numpy as np

class Singleton(type):
  _instance = {}
  
  def __call__(cls, *args, **kwargs):
    if cls not in Singleton, _instance:
      Singleton._instance[cls] = type.__call__(cls, *args, **kwargs)
    return Singleton._instance[cls]
  
# 通常参数服务器是一个很大的工程，这里简单使用内存实现一个比较low的参数服务
# 定义一个参数k-v结构map {hashcode, embedding}
class PS(metaclass = Singleton):
  def __init__(self, embedding_num):
    np.random.seed(2020)
    self.params_server = dict()
    self.dim = embeding_num
    print("ps inited...")
  
  def pull(self, keys):
    values = []
    # 这里传进来的数据是[batch, feature_len]
    for k in keys:
      tmp = []
      for arr in k:
        value = self.params_server.get(arr, None)
        if value is None:
          value = np.random.rand(self.dim)
          self.params_server[arr] = value
        tmp.append(value)
      values.append(tmp)
     return np.asarray(values, dtype = 'float32')
  
  def push(self, keys, values):
    for i in range(len(keys)):
      for j in range(len(keys[i])):
        self.params_server[keys[i][j]] = values[i][j]
```

```python
# 测试参数服务器
ps_local = PS(8)
keys = [[123, 234], [567, 891]]
# 从参数服务器pull keys，如果有直接取出，若没有就随机初始化并取出
res = ps_local.pull(keys)
print("参数服务器中有哪些参数: \n", ps_local.params_server)
print("keys, 获取到的对应向量: \n", res)

# 经过模型迭代更新后，传入参数服务器中
gradient = 10
res = res - 0.01 * gradient
ps_lcoal.push(keys, res)
print("经过迭代后，参数服务器中有哪些参数: \n", ps_local.params_server)

# 经过多轮pull参数，然后梯度更新后，保存最终的key对应的embedding
# 保存向量，该向量用户召回
path = "feature_embedding"
ps_loal.save(path)
```

##### 2.4.2 数据输入部分

读取tfrecords的数据，并从PS中取出对应的向量，构建完成的input层

```python
import tensorflow as tf
class InputFn:
  def __init__(self, local_ps):
    self.feature_len = 2 # 用户id，电影id
    self.label_len = 1
    self.n_parse_threads = 4
    self.shuffle_buffer_size = 1024
    self.prefetch_buffer_size = 1
    self.batch = 8
    self.local_ps = local_ps
   
  def input_fn(self, data_dir, is_test = False):
    # 解析tfrecord数据
    def _parse_example(example):
  		features = {
    		"feature_embedding": tf.io.FixedLenFeature(self.feature_len, tf.int64),
    		"label": tf.io.FixedLenFeature(self.label_len, tf.float32)
  		}
  		return tf.io.parse_single_example(example, features)
    
    def _get_embedding(parsed):
      keys = parsed["feature"]
      #keys_array = ps_rec.pull(keys)
      keys_array = tf.compat.v1.py_func(self.local_ps.pull, [keys], tf.float32)
      result = {
        "feature": parsed["feature"],
        "label": parsed["label"],
        "feature_embedding": keys_array
      }
      return result
    
    file_list = os.listdir(data_dir)
    files = []
    
    for i in range(len(file_list)):
      files.append(os.path.join(data_dir, file_list[i]))
    
    dataset = tf.compat.v1.data.Dataset.list_files(files)
    # 数据复制多少份
    if is_test:
      dataset = dataset.repeat(1)
    else:
      dataset = dataset.repeat()
      
    # 读取tfrecord数据
    dataset = dataset.interleave(
    			lambda_: tf.compat.v1.data.TFRecordDataset(_),
    			cycle_length = 1
    )
    
    # 对tfrecord的数据进行解析
    dataset = dataset.map(
    			_parse_example,
   				num_parallel_calls = self.n_parse_threads)
    
    # batch data,数据分批
    dataset = dataset.batch(
    			self.batch, drop_remainer = True)
    
    # get embedding
    dataset = dataset.map(
    			_get_embedding,
   				num_parallel_calls = self.n_parse_threads)
    
    # 对数据进行打乱
    if not is_test:
      dataset.shuffle(self.shuffle_buffer_size)
      
    # 数据与加载
    dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)
    
    # 迭代器
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    return iterator, iterator.get_next()
```

```python
# 测试数据输入层
local_ps = PS(8)
inputs = InputFn(local_ps)
data_dir = "./data/train"
train_iterator, train_inputs = inputs.input_fn(data_dir, is_test = False)

with tf.compat.v1.Session() as session:
  session.run(train_iterator.initializer)
  for i in range(1):
    print(session.run(train_inputs))
```

##### 2.4.3 模型实现部分

核心代码如下

```python
def mf_fn(inputs):
	# 取特征和y值，feature为user_id和movie_id
  embed_layer = inputs["feature_embedding"] # [batch, 2, embedding_dim]
  label = inputs["label"] #[batch, 1]
  # 切分数据，获得user_id的embedding和movie_id的embedding
 	embed_layer = tf.split(embed_layer, axis = 1) #[batch, embedding_dim] * 2
  user_id_embedding = embed_layer[0] #[batch, embedding_dim]
  movie_id_embedding = embed_layer[1]
  
  # 根据公示进行乘积并求和
  out_ = tf.reduce_mean(
  	user_id_embedding * movie_id_embedding, axis = 1) #[batch]
  
  # 损失函数loss
  label_ = tf.reshape(label, [-1]) #[batch]
  loss_  = tf.reduce_sum(tf.square(label_ = out_)) #1
```

详细代码如下

```python
import tensorflow as tf

batch = 32
embedding_dim = 8
learning_rate = 0.001

def mf_fn(inputs, is_test):
  # 取特征和y值，feature为user_id和movie_id
  embed_layer = inputs["feature_embedding"] #[batch, 2, embedding_dim]
  embed_layer = tf.reshape(embed_layer, shape = [-1, 2, embedding_dim])
  label = inputs["label"] #[batch, 1]
  # 切分数据，获得user id和movie_id的embedding
  embed_layer = tf.split(embed_layer, num_or_size_splits = 2, axis = 1)
  user_id_embedding  = tf.reshape(embed_layer[0], shape = [-1, embeding_dim]) #[batch, embedding]
  movie_id_embedding = tf.reshape(embed_layer[1], shape = [-1, embeding_dim])
  
  # 根据公式进行乘积并求和
  out_ = tf.reduce_mean(user_id_embedding * movie_embedding, axis = 1)
  
  # 设定预估部分
  out_tmp = tf.sigmoid(out_)
  
  if is_test:
    tf.compat.v1.add_to_collections("input_tensor" , embed_layer)
    tf.compat.v1.add_to_collections("output_tensor", out_tmp)
  # 损失函数loss
  label_ = tf.reshape(label, [-1])
  loss_  = tf.reduce_sum(tf.square(label_ - out_))
  
  out_dic = {
    "loss", loss_,
    "ground_truth": label_, #真实值
    "prediction": out_ #预估值
  }
  return out_dic

#定义整个图结果，并给出梯度更新方式
def setup_graph(inputs, is_test = False):
  result = {}
  with tf.compat.v1.variable_scope("net_graph", reuse = is_test):
    #初始化模型图
    net_out_dic = mf_fn(inputs, is_test)
    loss = net_out_dic["loss"]
    result["out"] = net_out_dic
    
    # 测试不用参数迭代
    if is_test:
      return result
    
    # SGD 随机梯度下降法，进行参数迭代
    embed_grad = tf.gradients(
    	loss, [inputs["feature_embedding"]], name = "feature_embedding")
    
    result["feature_new_embedding"] = \
    	inputs["feature_embedding"] - learning_rate * embed_grad
    
    result["feature_embedding"] = inputs["feature_embedding"]
    result["feature"] = inputs["feature"]
    
    return result
```

测试返回 ：{out:  {loss, ground_truth, prediction}}

训练返回：{out, feature_new_embedding, feature_embedding, feature}

##### 2.4.4 模型训练部分

训练模型主要考虑训练数据进行迭代，更新参数，然后设置迭代多少次进行一次测试集来测试效果

核心代码如下

```python
train_iterator, train_inputs = input_fn(train_file, is_test = False)
train_dic = setup_graph(train_inputs, is_test = False)

test_iterator, test_inputs = input_fn(test_file, is_test = True)
test_dic = setup_graph(test_inputs, is_test = True)

# 建立session，进行训练
with tf.compat.v1.Session() as session:
  # init global & local variables
  session.run([tf.compat.v1.global_variables_initializer(),
             tf.compat.v1.local_variables_initializer()])
	# 开始训练
  session.run(train_iterator.initializer)
  while _step < max_steps:
    new_ps_embed, keys, out = session.run()
```

使用sk-learn里的auc来检测效果

详细代码如下

```python
batch = 32
embedding_dim = 8
local_ps = PS(embedding_dim)
n_parse_threads = 4
shuffle_buffer_size = 1024
prefetch_buffer_size = 16
max_steps = 100000
test_show_step = 1000
# 数据输入
inputs = InputFn(local_ps)

last_test_auc = 0
# 训练评估
train_metric = AUCUtil()
test_metric = AUCUtil()

train_file = "./data/train"
test_file = "./data/val"
save_embedding = "./data/saved_embedding"

train_iterator, train_inputs = inputs.input_fn(train_file, is_test = False)
train_dic = setup_graph(train_inputs, is_test = False)

test_iterator, test_inputs = inputs.input_fn(test_file, is_test = True)
test_dic = setup_graph(test_inputs, is_test = True)

train_log_iter = 1000
last_test_auc = 0.5
def train():
  _step = 0
  # 建立session，并进行训练
  with tf.compat.v1.Sesstion() as session:
    # init global & local variables
    session.run([tf.compat.v1.global_variable_initializer(),
                tf.compat.v1.local_variable_initializer()])
    
    # 开始训练
    session.run(train_iterator.initializer)
    while _step < max_steps:
      feature_old_embedding, feature_new_embedding, keys, out = session.run(
      	[train_dic["feature_embedding"],
         train_dic["feature_new_embedding"],
         train_dic["feature"]
         train_dic["out"]]
      )
      
      train_metric.add(
        out["loss"],
        out["ground_truth"],
        out["prediction"]
      )
      
      # 更新ps参数服务
      local_ps.push(keys, feature_new_embedding)
      _step += 1
      
      # 每训练多少个batch的训练数据，就打印一次训练的aug等信息
      
      if _step % train_log_iter == 0:
        print("Train at setp %d: %s", _step, train_metric.calc_str())
        train_metric.reset()
        if _step % test_show_step == 0:
          valid_step(session, test_iterator, test_dic)

def valid_step(session, test_iterator, test_dic):
  test_metric.reset()
  session.run(test_iterator.initializer)
  global loast_test_auc
  while True:
    try:
      out = sesion.run(test_dic["out"])
      test_metric.add(
        out["loss"],
        out["ground_truth"],
        out["prediction"]
      )
    except tf.errors.OutofRangeError:
      print("Test at step : %s", test_metric.calc_str())
      if test_metric.calc()["auc"] > last_test_auc:
        last_test_auc = test_metric.calc()["auc"]
        local_ps.save(save_embedding)
      break
```

#### 2.6 结果部署方便召回服务调用

如何运用于召回？

- 基于一种向量相似方法计算向量之间的相似

- 取movie_id对应的embedding向量，计算每个movie_id相似的movie_ids，从而形成item2item的推荐模式

- 取所有user_id对应的embedding，与所有的movie_id计算相似后形成user2Item的推荐模式

##### 基于用户点击返回的召回策略

1. 取出向量里面的电影id对应的向量
2. 基于相似计算方法，算出每个电影id的相似电影topK，存入redis内
3. 线上基于用户的实时点击电影，召回与该电影相似的候选集

##### 基于用户直接关联相似电影的召回策略

1. 将用户id向量和电影id向量分别存入向量服务中
2. 线上用户id，基于用户id取出向量，然后拿该向量取检索相似的电影id，取topK的返回候选集

![RecallDemo](https://github.com/RTCFoundation/RecSys/blob/main/images/RecallDemo.jpg)



```python
# 读取模型文件
def read_embedding_file(file):
  dic = dict()
  with open(file) as f:
    for line in f:
      tmp = line.split("\t")
      embedding = [flot(_) for _ tmp[1].split(",")]
      dic[tmp[0]] = embeding
  return dic

#通过hash值获得原始值的方法
def get_hash2id(file):
  movie_dict = {}
  user_dict  = {}
  with open(file) as f:
    for line in f:
      tmp = line.split(",")
      movie_dict[str(bkdr2hash54("UserID=" + tmp[1]))] = tmp[1]
      user_dict[str(bkdr2hash54("MovieID=" + tmp[0]))] = tmp[0]
  return user_dict, movie_dict

def split_user_movie(embedding_file, train_file):
  user_dict, movie_dict = get_hash2id(train_file)
  embedding_dict = read_embedding_file(embedding_file)
  movie_embedding = {}
  user_embedding  = {}
  for k, v in embedding_dict.items():
    movie_id = movie_dict.get(k, None)
    if movie_id is not None:
      movie_embedding[movie_id] = v
    user_id = user_dict.get(k, None)
    if user_id is not None:
      user_embedding[user_id] = v
  return movie_embedding, user_embedding
```

```python
# 用于Item2Item模式
def col_sim(movie_sim_movie_file, movie_embeding):
  wfile = open(movie_sim_movie_file, "w")
  for m, vec1 in movie_embedding.items():
    sim_movie_tmp = {}
    for n, vec2 in movie_embedding.items():
      if m == n:
        continue
      sim_movie_tmp[n] = np.dot(np.asarray(vec2), np.asarray(vec1))
    
    sim_movie = sorted(sim_movie_tmp.items(), key = lambda _: _[1], reverse = True)
    sim_movie = [str(_[0]) for _ in sim_movie][:200]
    
    wfile.write(m + "\t" + ",".join(sim_movie) + "\n")
  
  wfile.close()
```

```python
# 用于user2Item模式
def write_uer_movie_embedding(movie_embeding_file, user_embedding_file, movie_embedding, user_embedding):
  wfile1 = open(movie_embedding_file, "w")
  for k, v in movie_embedding.items():
    wfile1.write(k + "\t" + ",".join([str(_) for _ in v]) + "\n")
 	wfile1.close()
  
  wfile2 = open(user_embedding_file, "w")
  for k, v in user_embedding.items():
    wfile2.write(k + "\t" + ",".join([str(_) for _ in v]) + "\n")
 	wfile2.close()
```

```python
# 测试
embedding_file = "./data/saved_embedding"
train_file = "./data/train_set"
movie_embedding, user_embedding = split_user_movie(embedding_file, train_file)

# 用于Item2Item模式
movie_sim_movie_file = "./data/movie_sim_movie_file"
col_sim(movie_sim_movie_file, movie_embedding)

# 用于User2item模式
movie_embedding_file = "./data/movie_embedding_file"
user_embedding_file  = "./data/user_embedding_file"
write_user_movie_embedding(movie_embdding_file, user_embedding_file, movie_embedding, user_embedding)
```

### 3. MatrixCF运用于排序

步骤：

a. 通常是对应的向量不用存储其他地方，直接从参数服务器中实时获取

b. 线上一个用户和对应的召回候选集过来，基于user_id取参数服务器中的用户向量，基于召回集里的电影id从参数服务器中取出电影的向量

c.基于相似计算的策略计算出用户与这些电影的分值，并基于分值进行倒排，取top K返回

缺点：

- 新用户和新电影是没有向量的

- 用户的行为是实时改变的，而向量被没有捕获到这个行为

为什么召回不存在这个问题

每路召回都是基于召回结果的一部分，可以使用其他策略来弥补这个问题
