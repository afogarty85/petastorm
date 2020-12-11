# Databricks notebook source
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, FloatType
from pyspark.sql import SQLContext, SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import hour, minute, second, to_timestamp, monotonically_increasing_id, row_number, lit, pow, percent_rank
from pyspark.sql.window import Window

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

#from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml import Pipeline

from petastorm.spark.spark_dataset_converter import _convert_vector
from petastorm.pytorch import DataLoader, BatchedDataLoader
from petastorm import make_batch_reader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
#from torch.utils.data.distributed import DistributedSampler


# for distributed computing
import horovod.torch as hvd
from sparkdl import HorovodRunner
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

# set number of cores; db says use less
spark.conf.set("spark.sql.shuffle.partitions", "120")

# enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "200")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# number of workers
sc._jsc.sc().getExecutorMemoryStatus().size()

# COMMAND ----------

# DBTITLE 1,Setup Checkpoints
PYTORCH_DIR = '/dbfs/ml/horovod_pytorch/take2'
 
LOG_DIR = os.path.join(PYTORCH_DIR, 'PetaFlights')
if os.path.exists(LOG_DIR) == False:
    os.makedirs(LOG_DIR)
    
def save_checkpoint(model, optimizer, epoch):
  filepath = LOG_DIR + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)

# COMMAND ----------

# load from view
df = spark.sql("select * from flights_all_v5")

# filter
df = df.filter(f.col('DEP_DELAY') >= -15)

# COMMAND ----------

# create unique tail
df = df.withColumn('unique_tail', f.concat(f.col("TAIL_NUM"), lit(" "), f.col("OP_UNIQUE_CARRIER")))

# COMMAND ----------

# create time conditions
w1 = Window.partitionBy().orderBy('unique_tail', 'departure_time')
w2 = Window.partitionBy('case_id').orderBy('unique_tail', 'departure_time')

df = df.withColumn("case_id", f.sum(f.when(~(f.col("unique_tail") == f.lag("unique_tail").over(w1)) | (f.lag("DEP_DEL15",1,0).over(w1) == 1),1).otherwise(0)).over(w1)+1) \
    .withColumn('time', f.count('*').over(w2)-1)

# create time polynomial
df = df.withColumn('time2', f.pow(f.col("time"), 2))
df = df.withColumn('time3', f.pow(f.col("time"), 3))

# sort by date
df = df.orderBy('unique_tail', 'departure_time', ascending=True)

# COMMAND ----------

# target, features
df = df.select('DEP_DEL15', "CRS_DEP_TIME", "DISTANCE", 'vis_distance', 'tmp', 'dew', 'elevation', 'dest_wnd_speed', 'pagerank', 'pagerank_dest', 'wnd_speed', 'cig_height', 'dest_vis_distance', 'dest_tmp', 'dest_dew', 'dest_elevation', 'dest_cig_height', 'departure_time', 'DEST_STATE_ABR', 'wnd_direction', 'dest_wnd_direction', 'OP_UNIQUE_CARRIER','OP_CARRIER', 'ORIGIN', 'ORIGIN_STATE_ABR', 'DEST', 'cig_code','cig_cavok_code','dest_cig_code','dest_cig_cavok_code','dest_vis_var_code', 'unique_tail', 'DAY_OF_WEEK', 'DEP_DEL15_PREV', 'MONTH', 'QUARTER', 'DAY_OF_MONTH', 'OP_CARRIER_AIRLINE_ID','OP_CARRIER_FL_NUM', 'DISTANCE_GROUP', 'OD_GROUP')

# drop some problematic NAs
df = df.na.drop(subset=["DEP_DEL15", 'unique_tail'])

# limit for small batch testing
#df = df.limit(130459)

# create a variable that will be used to split the data into train/valid/test later
df = df.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("departure_time")))

# COMMAND ----------

# DBTITLE 1,Convert String and Int. Categorical to Zero-min Categorical
# list of str features
str_features = ['DEST_STATE_ABR', 'wnd_direction', 'dest_wnd_direction','OP_UNIQUE_CARRIER','OP_CARRIER', 'ORIGIN', 'ORIGIN_STATE_ABR', 'DEST', 'cig_code','cig_cavok_code','dest_cig_code','dest_cig_cavok_code','dest_vis_var_code', 'unique_tail']

# list of int categorical features
int_categorical = ['DAY_OF_WEEK', 'DEP_DEL15_PREV', 'MONTH', 'QUARTER', 'DAY_OF_MONTH', 'OP_CARRIER_AIRLINE_ID','OP_CARRIER_FL_NUM', 'DISTANCE_GROUP', 'OD_GROUP']

# create indexers
indexers = [StringIndexer(inputCol=column, outputCol=column+"_cat").fit(df) for column in str_features+int_categorical]

# pipeline them
pipeline = Pipeline(stages=indexers)

# transform -- drop str and int features listed above
df = pipeline.fit(df).transform(df).drop('DEST_STATE_ABR', 'wnd_direction', 'dest_wnd_direction','OP_UNIQUE_CARRIER','OP_CARRIER', 'ORIGIN', 'ORIGIN_STATE_ABR', 'DEST', 'cig_code','cig_cavok_code','dest_cig_code','dest_cig_cavok_code','dest_vis_var_code', 'unique_tail', 'DAY_OF_WEEK', 'DEP_DEL15_PREV', 'MONTH', 'QUARTER', 'DAY_OF_MONTH', 'OP_CARRIER_AIRLINE_ID','OP_CARRIER_FL_NUM', 'DISTANCE_GROUP')

# COMMAND ----------

# DBTITLE 1,Continuous Features
cont_features = ["CRS_DEP_TIME", "DISTANCE", 'vis_distance', 'tmp', 'dew', 'elevation', 'dest_wnd_speed', 'pagerank', 'pagerank_dest', 'wnd_speed', 'cig_height', 'dest_vis_distance', 'dest_tmp', 'dest_dew', 'dest_elevation', 'dest_cig_height']

# COMMAND ----------

# DBTITLE 1,Categorical Features
cat_features = ['DEST_STATE_ABR_cat', 'wnd_direction_cat', 'dest_wnd_direction_cat','OP_UNIQUE_CARRIER_cat','OP_CARRIER_cat', 'ORIGIN_cat', 'ORIGIN_STATE_ABR_cat', 'DEST_cat', 'cig_code_cat','cig_cavok_code_cat','dest_cig_code_cat','dest_cig_cavok_code_cat','dest_vis_var_code_cat', 'unique_tail_cat', 'DAY_OF_WEEK_cat', 'DEP_DEL15_PREV_cat', 'MONTH_cat', 'QUARTER_cat', 'DAY_OF_MONTH_cat', 'OP_CARRIER_AIRLINE_ID_cat','OP_CARRIER_FL_NUM_cat', 'DISTANCE_GROUP_cat']

# COMMAND ----------

# DBTITLE 1,Create Time Series Train / Valid Sets
train_df = df.where("rank <= .8").drop("rank", "departure_time")
val_df = df.where("rank > .8 and rank < .9").drop("rank", "departure_time")
test_df = df.where("rank >= .9").drop("rank", "departure_time")

# COMMAND ----------

# DBTITLE 1,Assemble Continuous Featuers
# select features
assembler = VectorAssembler(inputCols=cont_features, outputCol="features")

# create vector train_df
assembled_train = assembler.transform(train_df).drop("CRS_DEP_TIME", "DISTANCE", 'vis_distance', 'tmp', 'dew', 'elevation', 'dest_wnd_speed', 'pagerank', 'pagerank_dest', 'wnd_speed', 'cig_height', 'dest_vis_distance', 'dest_tmp', 'dest_dew', 'dest_elevation', 'dest_cig_height')

# create vector val_df
assembled_val = assembler.transform(val_df).drop("CRS_DEP_TIME", "DISTANCE", 'vis_distance', 'tmp', 'dew', 'elevation', 'dest_wnd_speed', 'pagerank', 'pagerank_dest', 'wnd_speed', 'cig_height', 'dest_vis_distance', 'dest_tmp', 'dest_dew', 'dest_elevation', 'dest_cig_height')

# create vector test_df
assembled_test = assembler.transform(test_df).drop("CRS_DEP_TIME", "DISTANCE", 'vis_distance', 'tmp', 'dew', 'elevation', 'dest_wnd_speed', 'pagerank', 'pagerank_dest', 'wnd_speed', 'cig_height', 'dest_vis_distance', 'dest_tmp', 'dest_dew', 'dest_elevation', 'dest_cig_height')

# COMMAND ----------

# DBTITLE 1,Scale Continuous Features
# scale train
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True).fit(assembled_train)
assembled_train = scaler.transform(assembled_train).drop('features')
assembled_train = _convert_vector(assembled_train, 'float32')

# scale val 
assembled_val = scaler.transform(assembled_val).drop('features')
assembled_val = _convert_vector(assembled_val, 'float32')

# scale test 
assembled_test = scaler.transform(assembled_test).drop('features')
assembled_test = _convert_vector(assembled_test, 'float32')

# COMMAND ----------

# check partition size
assembled_val.rdd.getNumPartitions()

# COMMAND ----------

# DBTITLE 1,Write Data to Parquet
# write train to parquet; at least to # of workers
# write to dbfs/ml for extra speed performance
assembled_train.repartition(10) \
  .write.mode("overwrite") \
  .option("parquet.block.size", 1024 * 1024) \
  .parquet('file:///dbfs/ml/tmp/assembled_t')

# write val to parquet
assembled_val.repartition(10) \
  .write.mode("overwrite") \
  .option("parquet.block.size", 1024 * 1024) \
  .parquet('file:///dbfs/ml/tmp/assembled_v')

# write test to parquet
assembled_test.repartition(10) \
  .write.mode("overwrite") \
  .option("parquet.block.size", 1024 * 1024) \
  .parquet('file:///dbfs/ml/tmp/assembled_test')

# COMMAND ----------

# DBTITLE 1,Generate Embedding Data - Categorical Dimensions
# get counts of distinct features
fe = []
for v in cat_features:
  fe.append((v, df.select(v).distinct().count()))

# just get the cardinality
cat_dims = [x[1] for x in fe]

# find a general value for each
emb_dims = [(x, min(50, (x + 2) // 2)) for x in cat_dims]

# create embedding dict
embeddings = {}
for i, j in zip(fe, emb_dims):
  if i[0] not in embeddings:
    embeddings[i[0]] = j

# set embedding table shape for later use
embedding_table_shapes = embeddings

# COMMAND ----------

# show embedding dims
embeddings

# COMMAND ----------

# DBTITLE 1,Establish Embeddings
class ConcatenatedEmbeddings(torch.nn.Module):
    """Map multiple categorical variables to concatenated embeddings.
    Args:
        embedding_table_shapes: A dictionary mapping column names to
            (cardinality, embedding_size) tuples.
        dropout: A float.
    Inputs:
        x: An int64 Tensor with shape [batch_size, num_variables].
    Outputs:
        A Float Tensor with shape [batch_size, embedding_size_after_concat].
    """

    def __init__(self, embedding_table_shapes, dropout=0.2):
        super().__init__()
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(cat_size, emb_size)
                for cat_size, emb_size in embedding_table_shapes.values()
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = [layer(x[:, i]) for i, layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return x

# COMMAND ----------

# DBTITLE 1,MLP
class FF_NN(torch.nn.Module):
    def __init__(self, num_features, num_classes, drop_prob, embedding_table_shapes, num_continuous, emb_dropout):
        # deep NN with batch norm
        super(FF_NN, self).__init__()
        # first hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # batch norm
        self.linear_1_bn = torch.nn.BatchNorm1d(num_hidden_1)
        # second hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        # batch norm
        self.linear_2_bn = torch.nn.BatchNorm1d(num_hidden_2)
        # third hidden layer
        self.linear_3 = torch.nn.Linear(num_hidden_2, num_hidden_3)
        # batch norm
        self.linear_3_bn = torch.nn.BatchNorm1d(num_hidden_3)        
        # output layer
        self.linear_out = torch.nn.Linear(num_hidden_3, num_classes)
        # dropout
        self.drop_prob = drop_prob
        # cat
        self.initial_cat_layer = ConcatenatedEmbeddings(embedding_table_shapes, dropout=emb_dropout)
        # cont
        self.initial_cont_layer = torch.nn.BatchNorm1d(num_continuous)
 
    # define how and what order model parameters should be used in forward prop.
    def forward(self, x_cat, x_cont):
        x_cat = self.initial_cat_layer(x_cat)
        x_cont = self.initial_cont_layer(x_cont)
        x = torch.cat([x_cat, x_cont], 1)
        # run inputs through first layer
        out = self.linear_1(x)
        # apply dropout -- doesnt matter position with relu
        out = F.dropout(out, p=self.drop_prob, training=self.training)   
        # apply relu
        out = F.relu(out)
        # apply batchnorm
        out = self.linear_1_bn(out)        
        # run inputs through second layer
        out = self.linear_2(out)
        # apply dropout -- doesnt matter position with relu
        out = F.dropout(out, p=self.drop_prob, training=self.training)        
        # apply relu
        out = F.relu(out)
        # apply batchnorm
        out = self.linear_2_bn(out)        
        # run inputs through third layer
        out = self.linear_3(out)
        # apply dropout -- doesnt matter position with relu
        out = F.dropout(out, p=self.drop_prob, training=self.training)        
        # apply relu
        out = F.relu(out)
        # apply batchnorm
        out = self.linear_3_bn(out)          
        # run inputs through final classification layer
        logits = self.linear_out(out)
        probas = F.log_softmax(logits, dim=1)
        return logits, probas
        
# load the NN model
num_hidden_1 = 1000
num_hidden_2 = 1000
num_hidden_3 = 1000
drop_prob = 0.3
num_classes = 2
emb_dropout = 0.1
num_continuous = len(cont_features)
num_features = sum(emb_size for _, emb_size in embedding_table_shapes.values()) + num_continuous
 
model = FF_NN(num_features=num_features, num_classes=2, drop_prob=drop_prob, embedding_table_shapes=embeddings, num_continuous=num_continuous, emb_dropout=emb_dropout)

# COMMAND ----------

# show num features
print(num_features)

# COMMAND ----------

# DBTITLE 1,Train Metrics
# train metrics
train_df_size = train_df.count()
val_df_size = val_df.count()
test_df_size = test_df.count()
print(train_df_size)
print(val_df_size)
print(test_df_size)

# COMMAND ----------

train_df.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

# n samples / n_classes * bincount
print(((20080658+4473741) / (2 * np.array([20080658, 4473741]))))

BATCH_SIZE = 100
NUM_EPOCHS = 7
weighting = torch.tensor([0.61139428, 2.74428035])  # impose higher costs on misclassified 1s

# COMMAND ----------

# DBTITLE 1,Transform Data
def _transform_row(batch, cont_cols=['scaledFeatures'], cat_cols=cat_features, label_cols=['DEP_DEL15']):
    x_cat, x_cont, y = None, None, None
    x_cat = [batch[col].type(torch.LongTensor) for col in cat_cols]
    x_cat = torch.stack(x_cat, 1)
    x_cont = batch['scaledFeatures']
    y = batch['DEP_DEL15']
    return x_cat.to(device), x_cont.to(device), y.to(device)

# COMMAND ----------

# DBTITLE 1,Check Data
train_loader = BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/ml/tmp/assembled_t', num_epochs=None,
                                                   transform_spec=None,
                                                   shuffle_row_groups=False,
                                                  workers_count=8), batch_size=4)
x_cat, x_cont, y = _transform_row(next(iter(train_loader)))
print(x_cat, x_cont.squeeze(1), y)

# COMMAND ----------

# DBTITLE 1,One Epoch Loop
def train_one_epoch(model, optimizer, scheduler, 
                    train_dataloader_iter, steps_per_epoch, epoch, 
                    device):
  model.train()  # Set model to training mode

  # statistics
  running_loss = 0.0
  running_corrects = 0
  
  # Iterate over the data for one epoch.
  for step in range(steps_per_epoch):
    x_cat, x_cont, labels = _transform_row(next(train_dataloader_iter))
    
    # Track history in training
    with torch.set_grad_enabled(True):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      logits, probas = model(x_cat.long(), x_cont.squeeze(1))
      _, preds = torch.max(probas, 1)
      loss = F.cross_entropy(logits, labels.long(), weight=weighting)

      # backward + optimize
      loss.backward()
      optimizer.step()

    # statistics
    running_loss += loss.item() * x_cat.size(0)
    running_corrects += torch.sum(preds == labels)
  
  scheduler.step()

  epoch_loss = running_loss / (steps_per_epoch * BATCH_SIZE)
  epoch_acc = running_corrects.double() / (steps_per_epoch * BATCH_SIZE)

  print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

def evaluate(model, val_dataloader_iter, validation_steps, device, 
             metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over all the validation data.
  for step in range(validation_steps):
    x_cat, x_cont, labels = _transform_row(next(val_dataloader_iter))

    # Do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      # forward
      logits, probas = model(x_cat.long(), x_cont.squeeze(1))
      _, preds = torch.max(probas, 1)
      loss = F.cross_entropy(logits, labels.long(), weight=weighting)
      
    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels)
   
  # Average the losses across observations for each minibatch.
  epoch_loss = running_loss / validation_steps
  epoch_acc = running_corrects.double() / (validation_steps * BATCH_SIZE)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')

  print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

# COMMAND ----------

# DBTITLE 1,Single Worker
def train_and_evaluate(lr=0.016):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = FF_NN(num_features=num_features, num_classes=2, drop_prob=drop_prob, embedding_table_shapes=embeddings, num_continuous=num_continuous, emb_dropout=emb_dropout)
  
  # Only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1)

  # Decay LR by a factor of 0.1 every 3 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
  
  with BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/ml/tmp/assembled_t',
                                    num_epochs=None, transform_spec=None, shuffle_row_groups=False,
                                           workers_count=8), batch_size=BATCH_SIZE) as train_dataloader, \
       BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/ml/tmp/assembled_v',
                                    num_epochs=None, transform_spec=None, shuffle_row_groups=False,
                                           workers_count=8), batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = train_df_size // BATCH_SIZE
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = validation_steps = val_df_size // BATCH_SIZE
    
    for epoch in range(NUM_EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train_one_epoch(model, optimizer, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      
      val_loss, val_acc = evaluate(model, val_dataloader_iter, validation_steps, device)
      
  return val_loss, val_acc
  
#loss = train_and_evaluate()

# COMMAND ----------

# DBTITLE 1,Horovod
def metric_average(val, name):
  tensor = torch.as_tensor(val)
  avg_tensor = hvd.allreduce(tensor, name=name)
  return avg_tensor.item()

def train_and_evaluate_hvd(lr=0.016):
  hvd.init()  # Initialize Horovod.
  
  # Horovod: pin GPU to local rank.
  if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    device = torch.cuda.current_device()
  else:
    device = torch.device("cpu")
  
  model = FF_NN(num_features=num_features, num_classes=2, drop_prob=drop_prob, embedding_table_shapes=embeddings, num_continuous=num_continuous, emb_dropout=emb_dropout)

  # Effective batch size in synchronous distributed training is scaled by the number of workers.
  # An increase in learning rate compensates for the increased batch size.
  optimizer = torch.optim.SGD(model.parameters(), lr=lr * hvd.size(), momentum=0.9)
  
  # Broadcast initial parameters so all workers start with the same parameters.
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  
  # Wrap the optimizer with Horovod's DistributedOptimizer.
  optimizer_hvd = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hvd, step_size=5, gamma=0.1)

  with BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/ml/tmp/assembled_t',
                                    num_epochs=None, cur_shard=hvd.rank(), shard_count=hvd.size(),
                                    transform_spec=None, shuffle_row_groups=False, workers_count=8), batch_size=BATCH_SIZE) as train_dataloader, \
       BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/ml/tmp/assembled_v',
                                    num_epochs=None, cur_shard=hvd.rank(), shard_count=hvd.size(),
                                    transform_spec=None, shuffle_row_groups=False, workers_count=8), batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = train_df_size // (BATCH_SIZE * hvd.size())
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps =  val_df_size // (BATCH_SIZE * hvd.size())
    
    for epoch in range(NUM_EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train_one_epoch(model, optimizer_hvd, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      
      # save checkpoint
      if hvd.rank() == 0: save_checkpoint(model, optimizer_hvd, epoch)
      
      val_loss, val_acc = evaluate(model, val_dataloader_iter, validation_steps,
                                   device, metric_agg_fn=metric_average)
      
  return val_loss, val_acc

# COMMAND ----------

hr = HorovodRunner(np=10)   # This assumes the cluster consists of 10 workers.
hr.run(train_and_evaluate_hvd)

# COMMAND ----------

# DBTITLE 1,Review Checkpoint Files
# review checkpoint files
display(dbutils.fs.ls('dbfs:/ml/horovod_pytorch/take2/PetaFlights'))

# COMMAND ----------

# DBTITLE 1,Single Worker Test Set
NUM_EPOCHS=1
BATCH_SIZE=100

def evaluate(model, val_dataloader_iter, validation_steps, device, 
             metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0
  
  # for f1 and other metrics
  global preds1
  preds1 = []
  global labels1
  labels1 = []
  
  # Iterate over all the validation data.
  for step in range(validation_steps):
    x_cat, x_cont, labels = _transform_row(next(val_dataloader_iter))

    # Do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      # forward
      logits, probas = model(x_cat.long(), x_cont.squeeze(1))
      _, preds = torch.max(probas, 1)
      loss = F.cross_entropy(logits, labels.long(), weight=weighting)
      
      preds1.append(preds)
      labels1.append(labels)

    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels)
   
  # Average the losses across observations for each minibatch.
  epoch_loss = running_loss / validation_steps
  epoch_acc = running_corrects.double() / (validation_steps * BATCH_SIZE)
  epoch_f1 = f1_score(torch.cat(preds1), torch.cat(labels1), average='weighted')
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')

  print('Test Loss: {:.4f} Acc: {:.4f}, F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
  return epoch_loss, epoch_acc, epoch_f1

def train_and_evaluate(lr=0.016):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = FF_NN(num_features=num_features, num_classes=2, drop_prob=drop_prob, embedding_table_shapes=embeddings, num_continuous=num_continuous, emb_dropout=emb_dropout)
  model.load_state_dict(torch.load('/dbfs/ml/horovod_pytorch/take2/PetaFlights/checkpoint-2.pth.tar')['model'])
  
  # Only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1)

  # Decay LR by a factor of 0.1 every 3 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
  
  with BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/ml/tmp/assembled_test',
                                    num_epochs=None,
                                           transform_spec=None,
                                           shuffle_row_groups=False),
                         batch_size=BATCH_SIZE) as val_dataloader:
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = val_df_size // BATCH_SIZE
    
    for epoch in range(NUM_EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)

      val_loss, val_acc, val_f1 = evaluate(model, val_dataloader_iter, validation_steps, device)

  return val_loss, val_acc, val_f1
  
loss, acc, f1 = train_and_evaluate()

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
print(confusion_matrix(torch.cat(preds1), torch.cat(labels1)))

# COMMAND ----------

# DBTITLE 1,Classification Report
print(classification_report(torch.cat(preds1), torch.cat(labels1)))

# COMMAND ----------

# DBTITLE 1,ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(torch.cat(preds1), torch.cat(labels1), average='weighted')

# COMMAND ----------

# DBTITLE 1,F1 Macro
f1_score(torch.cat(preds1), torch.cat(labels1), average='weighted')

# COMMAND ----------

# intentionally left blank

# COMMAND ----------

# DBTITLE 1,Hyperparameter Search: Distributed Hyperopt
def train_and_evaluate(lr=0.001, weight_decay=2, batch_size=BATCH_SIZE):
  hvd.init()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = FF_NN(num_features=num_features, num_classes=2, drop_prob=drop_prob, embedding_table_shapes=embeddings, num_continuous=num_continuous, emb_dropout=emb_dropout)
  criterion = torch.nn.CrossEntropyLoss()

  # Only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  with BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/tmp/assembled_t',
                                    num_epochs=None,
                                    transform_spec=None, shuffle_row_groups=False, workers_count=8,
                                          cur_shard=hvd.rank(), shard_count=hvd.size()), batch_size=BATCH_SIZE) as train_dataloader, \
       BatchedDataLoader(make_batch_reader(dataset_url_or_urls='file:///dbfs/tmp/assembled_v',
                                    num_epochs=None, transform_spec=None, shuffle_row_groups=False, workers_count=8,
                                          cur_shard=hvd.rank(), shard_count=hvd.size()), batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = train_df_size // BATCH_SIZE
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps =  max(1, val_df_size // (BATCH_SIZE))
    
    for epoch in range(NUM_EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train_one_epoch(model, optimizer, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      val_loss, val_acc, val_f1 = evaluate(model, val_dataloader_iter, validation_steps, device)

  return val_loss

# COMMAND ----------

# DBTITLE 1,Hyperopt
BATCH_SIZE=100
NUM_EPOCHS=1
def train_fn(lr):
  loss = train_and_evaluate(lr)
  return {'loss': loss, 'status': STATUS_OK}

search_space = hp.loguniform('lr', -10, -4)

argmin = fmin(
  fn=train_fn,
  space=search_space,
  algo=tpe.suggest,
  max_evals=1,
  trials=SparkTrials(parallelism=8))

# COMMAND ----------

argmin

# COMMAND ----------


