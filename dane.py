# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import logging
import sys
import math
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count

logger = logging.getLogger("deepwalk") #获取日志实例


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
number = [];

class Graph(defaultdict): 
  """defaultydict 可以设置初始值 减少访问不存在key时错误发生"""
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list) #构建的图是一个字典，key是节点，key对应的value是list

  def node_int_value(self, node, flag):
    global number
    try:
      int_value = number.index(node)
    except ValueError:
      if flag == True:
        int_value = len(number)
        number.append(node)
      else:
        int_value = -1
    return int_value

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems() #返回本身字典列表操作后的迭代

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes] #nodes表示收缩节点范围
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent() #排序、去重、除掉自循环
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return len(self)

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
        这里的随机游走路径未必是连续的，有可能是走着走着突然回到起点接着走
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0]) #有概率从头开始
      else:
        break
    return [str(node) for node in path]
  
  def ToMatrix(self, dimen):
    lists = [[] for i in range(len(number))]
    for i in range(len(number)):
      data = [0] * dimen
      for j in self[i]:
        data[j] = 1
      lists[i]=data
    return lists
    

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  """
    这个函数可以对一个图生成一个语料库
    :param num_paths: 路径数量
    """

  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1))) ###############3#############


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()

  with open(file_) as f:
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      total = 0 
      for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
          adjlist.extend(adj_chunk)
          total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = G.node_int_value(x, False)
      y = G.node_int_value(y, False)
      if x != -1 & y != -1:
        G[y].append(x)
        if undirected:
            G[x].append(y)
  
  #G.make_consistent()
  return G

def load_citeseer_content(file):
  G = Graph()
  with open(file) as f:
    for l in f:
      line = l.strip().split()
      paper_id = G.node_int_value(line[0], True)
      line = line[1:]
      for i in range(len(line)):
        if line[i] == '1':
          G[paper_id].append(i)
        elif line[i] == '0':
          continue
        else:
          G[paper_id].append(i + {
            'Agents': 0,
            'AI': 1,
            'DB': 2,
            'IR': 3,
            'ML': 4,
            'HCI': 5
          }.get(line[i], 0))
  return G

def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10

n_input_citeseer_cities = 2321
n_input_citeseer_content = 3709
X = tf.placeholder("float")
X_content = tf.placeholder("float")

n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_hidden_3 = 256 # 1st layer num features
n_hidden_4 = 128 # 2nd layer num features
n_hidden_5 = 64 # 3nd layer num features
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input_citeseer_cities, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_1, n_input_citeseer_cities])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b5': tf.Variable(tf.random_normal([n_input_citeseer_cities])),
}

weights_content = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input_citeseer_content, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_1, n_input_citeseer_content])),
}
biases_content = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b5': tf.Variable(tf.random_normal([n_input_citeseer_content])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
                                   # Decoder Hidden layer with sigmoid activation #2
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                  biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                  biases['encoder_b4']))
    layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']),
                                   biases['encoder_b5'])
    return layer_5


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
                                   biases['decoder_b5']))
    return layer_5

# Building the encoder
def encoder_content(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_content['encoder_h1']),
                                   biases_content['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_content['encoder_h2']),
                                   biases_content['encoder_b2']))
                                   # Decoder Hidden layer with sigmoid activation #2
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_content['encoder_h3']),
                                  biases_content['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights_content['encoder_h4']),
                                  biases_content['encoder_b4']))
    layer_5 = tf.add(tf.matmul(layer_4, weights_content['encoder_h5']),
                                   biases_content['encoder_b5'])
    return layer_5


# Building the decoder
def decoder_content(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_content['decoder_h1']),
                                   biases_content['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_content['decoder_h2']),
                                   biases_content['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_content['decoder_h3']),
                                   biases_content['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights_content['decoder_h4']),
                                   biases_content['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights_content['decoder_h5']),
                                   biases_content['decoder_b5']))
    return layer_5

def GetP(i, j, decodedM, decodedZ):
  sum_value = 0
  for k in range(len(decodedZ[i])):
      sum_value += decodedM[i][k] * decodedZ[j][k]
  p = 1 / (1 + math.exp(-1 * sum_value))

def GetFirstOrderP(lists, decoded):
    ans = tf.add(tf.constant([0.]), tf.constant([0.]))
    for i in range(len(lists)):
        for j in range(len(lists[i])):
            if lists[i][j] == 1:
                ans = tf.add(ans, tf.log(GetP(i, j, decoded, decoded)))
    return ans 

def GetConsistent(lists, decodedM, decodedZ):
  ans = tf.add(tf.constant([0.]), tf.constant([0.]))
  for i in range(len(lists)):
    ans = tf.add(ans, tf.log(GetP(i, i, decodedM, decodedZ)))
    for j in range(len(lists[i])):
      if lists[i][j] == 0:
        ans = tf.subtract(ans, tf.log(1 - GetP(i, i, decodedM, decodedZ)))
  return ans 

def AutoEncoder(input_data, intput_init_data, input_data_attribute):
    global number

    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    encoder_op_attribute = encoder_content(X_content)
    decoder_op_attribute = decoder_content(encoder_op_attribute)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Prediction
    y_pred_attribute = decoder_op_attribute
    # Targets (Labels) are the input data.
    y_true_attribute = X_content

    ans = [[] for i in range(len(number))]

    with tf.Session() as sess:
      if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
      else:
          init = tf.global_variables_initializer()
      sess.run(init)

      #first = tf.add(GetFirstOrderP(intput_init_data, encoder_op.eval().tolist()), GetFirstOrderP(intput_init_data, encoder_op_attribute.eval().tolist()))
      cost = tf.add(tf.reduce_mean(tf.pow(y_true - y_pred, 2)), tf.reduce_mean(tf.pow(y_true_attribute - y_pred_attribute, 2)))
      #ans = tf.subtract(second, first)
      #cost = tf.subtract(ans, GetConsistent(intput_init_data, encoder_op.eval().tolist(), encoder_op_attribute.eval().tolist()))
      optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

      total_batch = int(len(number)/batch_size)
    # Training cycle
      for epoch in range(training_epochs):
          # Loop over all batches
          for i in range(total_batch):
              batch_xs = input_data[batch_size * i : batch_size * i + batch_size]
              batch_xs_content = input_data_attribute[batch_size * i : batch_size * i + batch_size]
              # Run optimization op (backprop) and cost op (to get loss value)
              _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, X_content: batch_xs_content})
          # Display logs per epoch step
      encoder_result = sess.run(encoder_op, feed_dict={X: input_data})
      encoder_result1 = sess.run(encoder_op, encoder_op_attribute={X_content: batch_xs_content})
      for i in range(len(number)):
        ans[i].append(encoder_result[i])
        ans[i].append(encoder_result1[i])
      
    return ans

def GetTransitionMatrix(matrix, m):
  m1 = np.array(matrix)
  m2 = np.array(m)
  return np.dot(m2, m1)

def FromEToM(e, t):
  ans = GetTransitionMatrix(e, e)
  step = ans
  for i in range(1, t):
    step = GetTransitionMatrix(e, step.tolist())
    ans = ans + step
  return ans.tolist()

def OuFile(data):
  filename = '/data/citeseer/output.embeddings'
  with open(filename,'w') as f:
    for i in range(len(data)):
      f.write(number[i])
      f.write(" ")
      f.write(data[i])
      f.write("\n")
  f.close()


def process(inputfile, inputattributefile, outputfile):
    A = load_citeseer_content(inputattributefile)
    print("Number of attribute structure nodes: {}".format(len(A.nodes())))

    G = load_edgelist(inputfile, undirected=False)
    print("Number of topological structure nodes: {}".format(len(G.nodes())))

    e = G.ToMatrix(len(number))
    answer = AutoEncoder(FromEToM(e, 10), e, A.ToMatrix(n_input_citeseer_content))
    OuFile(answer.tolist())

def main():
    inputfile = "./data/citeseer/citeseer.cites"
    inputattributefile = "./data/citeseer/citeseer.content"
    outputfile = "./data/citeseer/output.embeddings"

    process(inputfile, inputattributefile, outputfile)

if __name__ == "__main__":
  sys.exit(main())