from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from collections import defaultdict, Iterable

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

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

A = load_citeseer_content("./data/citeseer/citeseer.content")
print("Number of attribute structure nodes: {}".format(len(A.nodes())))
G = A.ToMatrix(3709)

Edge_graph = load_edgelist("./data/citeseer/citeseer.cites", undirected=False)
One_hot = Edge_graph.ToMatrix(len(number))
M = FromEToM(One_hot, 10)
    
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

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
    layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']), biases['encoder_b5'])
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

def GetSpecificVector(encoder_data, num):
    t = [[0] * 64] * len(number)
    t[num] = [1] * 64 
    return tf.reduce_sum(tf.multiply(encoder_data, t), 0)

def GetFirstOrderP(encoder_data, i, j):
    Hi = GetSpecificVector(encoder_data, i)
    Hj = GetSpecificVector(encoder_data, j)
    return tf.div(float(1), tf.add(float(1), tf.exp(tf.negative(tf.multiply(Hi, tf.transpose(Hj))))))

def GetFirstOrder(encoder_data):
    ans = tf.constant(float(0))
    for i in range(len(One_hot)):
        for j in range(len(One_hot[i])):
            ans = tf.add(ans, tf.log(GetFirstOrderP(encoder_data, i, j)))
    return tf.negative(ans)

def GetConsistent(lists, decodedM, decodedZ):
    ans = tf.add(tf.constant([0.]), tf.constant([0.]))
    for i in range(len(lists)):
        ans = tf.add(ans, tf.log(GetP(i, i, decodedM, decodedZ)))
        for j in range(len(lists[i])):
            if lists[i][j] == 0:
                ans = tf.subtract(ans, tf.log(tf.subtract(1, GetP(i, i, decodedM, decodedZ))))
    return ans 

def OutputFile(data):
    filename = 'data/citeseer/output.embeddings'
    with open(filename,'w') as f:
        for i in range(len(data)):
            f.write(number[i])
            f.write(" ")
            for j in range(len(data[i])):
                f.write(str(data[i][j]))
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

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
loss = tf.add(loss, GetFirstOrder(encoder_op))
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
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.array(G[index: index + batch_size])
        index += batch_size
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: G})
        # Display logs per step
        #if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
    encoder_result = sess.run(encoder_op, feed_dict={X: G})
    OutputFile(encoder_result.tolist())
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