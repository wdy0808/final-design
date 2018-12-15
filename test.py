from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from collections import defaultdict, Iterable

number = [];

class Graph(defaultdict): 
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

    def ToMatrix(self, dimen):
        lists = [[] for i in range(len(number))]
        for i in range(len(number)):
            data = [0] * dimen
            for j in self[i]:
                data[j] = 1
            lists[i]=data
        return lists

    def nodes(self):
        return self.keys()

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

def GetTransitionMatrix(matrix, m):
    m1 = np.array(matrix)
    m2 = np.array(m)
    t = np.dot(m2, m1)
    return t

def FromEToM(e, t):
    ans = GetTransitionMatrix(e, e)
    step = ans
    for i in range(1, t):
        step = GetTransitionMatrix(e, step.tolist())
        ans = ans + step
    return ans.tolist()

def ReverseMatric(adj):
    reverse_metric = [[] for i in range(len(adj))]
    binary_metric = [[] for i in range(len(adj))]
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            if adj[i][j] == 1:
                reverse_metric[i].append(0.0)
                binary_metric[i].append(-1.0)
            else:
                reverse_metric[i].append(1.0)
                binary_metric[i].append(1.0)
    return reverse_metric, binary_metric

A = load_citeseer_content("./data/citeseer/citeseer.content")
print("Number of attribute structure nodes: {}".format(len(A.nodes())))
G = A.ToMatrix(3709)

Edge_graph = load_edgelist("./data/citeseer/citeseer.cites", undirected=True)
print("Number of topological structure nodes: {}".format(len(Edge_graph.nodes())))
One_hot = Edge_graph.ToMatrix(len(number))
One_hot_reverse, One_hot_binary = ReverseMatric(One_hot)
#M = FromEToM(One_hot, 1)
M = One_hot
print("Finish Geting High Order Matrix-M")
    
# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_hidden_3 = 256 # 1st layer num features
n_hidden_4 = 128 # 2nd layer num features
n_hidden_5 = 64 # 3nd layer num features
num_input = 3709 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
X1 = tf.placeholder("float", [None, len(number)])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_1, num_input])),
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
    'decoder_b5': tf.Variable(tf.random_normal([num_input])),
}

weights1 = {
    'encoder_h1': tf.Variable(tf.random_normal([len(number), n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_1, len(number)])),
}
biases1 = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b5': tf.Variable(tf.random_normal([len(number)])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['encoder_h5']), biases['encoder_b5']))
    return layer_5


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']), biases['decoder_b5']))
    return layer_5

# Building the encoder
def encoder1(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights1['encoder_h1']), biases1['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights1['encoder_h2']), biases1['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights1['encoder_h3']), biases1['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights1['encoder_h4']), biases1['encoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights1['encoder_h5']), biases1['encoder_b5']))
    return layer_5


# Building the decoder
def decoder1(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights1['decoder_h1']), biases1['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights1['decoder_h2']), biases1['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights1['decoder_h3']), biases1['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights1['decoder_h4']), biases1['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights1['decoder_h5']), biases1['decoder_b5']))
    return layer_5

def GetP(encoded1, encoded2):
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(tf.matmul(encoded1, tf.transpose(encoded2))))))

def GetFirstOrder(H):
    return tf.negative(tf.reduce_sum(tf.log(tf.add(tf.multiply(H, One_hot), One_hot_reverse))))

def GetConsistent(H):
    former = tf.reduce_sum(tf.log(H))
    later = tf.reduce_sum(tf.log(tf.subtract(tf.constant(1.0), tf.multiply(H, One_hot_reverse))))
    return later
    #return tf.subtract(later, former)

def GetCost(encodedM, encodedZ):
    Pmm = GetP(encodedM, encodedM)
    Pzz = GetP(encodedZ, encodedZ)
    Pmz = GetP(encodedM, encodedZ)

    first_order_cost_m = GetFirstOrder(Pmm)
    first_order_cost_z = GetFirstOrder(Pzz)
    
    first_order_cost = tf.add(first_order_cost_m, first_order_cost_z)

    #return first_order_cost
    return GetConsistent(Pmz)
    #return tf.add(first_order_cost, GetConsistent(Pmz))

def OutputFile(data, data1):
    filename = 'data/citeseer/output.embeddings'
    with open(filename,'w') as f:
        for i in range(len(data)):
            f.write(number[i])
            f.write(" ")
            for j in range(len(data[i])):
                f.write(str(data[i][j]))
                f.write(" ")
            for j in range(len(data1[i])):
                f.write(str(data1[i][j]))
                f.write(" ")
            f.write("\n")
    f.close()

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Construct model
encoder_op1 = encoder1(X1)
decoder_op1 = decoder1(encoder_op1)

# Prediction
y_pred1 = decoder_op1
# Targets (Labels) are the input data.
y_true1 = X1

# Define loss and optimizer, minimize the squared error
loss = tf.add(tf.reduce_mean(tf.pow(y_true - y_pred, 2)), tf.reduce_mean(tf.pow(y_true1 - y_pred1, 2)))
loss = tf.add(loss, GetCost(encoder_op1, encoder_op))
#Sloss = tf.add(loss, tf.reduce_mean([[1.0,2.0],[3.0,4.0]]))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    index = 0
    # Training
    num_steps = int(len(number) / batch_size)

    for i in range(1, 3):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.array(G[0: 3312])
        batch_x1 = np.array(M[0: 3312])
        index += batch_size
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, X1: batch_x1})
        # Display logs per step
        #if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
    encoder_result = sess.run(encoder_op, feed_dict={X: G})
    encoder_result1 = sess.run(encoder_op1, feed_dict={X1: M})
    OutputFile(encoder_result.tolist(), encoder_result1.tolist())
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    '''n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()'''