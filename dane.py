from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

#io, including generate input data and output file.............
number = []
number_of_nodes = 2708
number_of_attribute = 1433

def node_int_value(node, flag):
    try:
        int_value = number.index(node)
    except ValueError:
        if flag == True:
            int_value = len(number)
            number.append(node)
        else:
            int_value = -1
    return int_value

def load_edgelist(file_, undirected=True):
    map_edge = [[0] * number_of_nodes for i in range(number_of_nodes)]
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = node_int_value(x, False)
            y = node_int_value(y, False)
            if x == -1:
                continue
            if y == -1:
                continue
            map_edge[y][x] = 1
            if undirected:
                map_edge[x][y] = 1
    return map_edge

def load_cora_content(file):
    map_attribute = [[0] * number_of_attribute for i in range(number_of_nodes)]
    with open(file) as f:
        for l in f:
            line = l.strip().split()
            paper_id = node_int_value(line[0], True)
            line = line[1:]
            for i in range(len(line)):
                if line[i] == '1':
                    map_attribute[paper_id][i] = 1
    return map_attribute

def GetTransitionMatrix(matrix, m):
    t = np.dot(matrix, m)
    return t

def Normalization(matrix):
    for i in range(len(matrix)):
        amax, amin = matrix[i].max(), matrix[i].min()
        if (amax > amin):
            matrix[i] = (matrix[i] - amin) / (amax - amin)
    return matrix

def FromEToM(e, t):
    e = Normalization(np.array(e, dtype=float))
    ans = GetTransitionMatrix(e, e)
    step = ans
    for i in range(1, t):
        print(i)
        step = GetTransitionMatrix(e, step)
        ans = Normalization(ans + step)
    return ans.tolist()

def ReverseMatric(adj):
    reverse_metric = [[] for i in range(len(adj))]
    diagronal_matrix = [[] for i in range(len(adj))]
    for i in range(len(adj)):
        tem = [0] * len(adj[i])
        for j in range(len(adj[i])):
            if adj[i][j] == 1:
                reverse_metric[i].append(0.0)
            else:
                reverse_metric[i].append(1.0)
        tem[i] = 1
        diagronal_matrix[i] = tem
    return reverse_metric, diagronal_matrix

def OutputFile(data, data1):
    filename = 'data/cora/output.embeddings'
    with open(filename,'w') as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                f.write(str(data[i][j]))
                f.write(" ")
            for j in range(len(data1[i])):
                f.write(str(data1[i][j]))
                f.write(" ")
            f.write("\n")
    f.close()

matrix_attribute = load_cora_content("./data/cora/cora.content")
print("Input attribute data done")

matrix_topology = load_edgelist("./data/cora/cora.cites", undirected=False)
print("Input topological structure data done")
matrix_topology_reverse, Diagonal_matrix = ReverseMatric(matrix_topology)
matrix_attribute_high_order = FromEToM(matrix_topology, 10)
print("Finish Geting High Order Matrix-M")
    
#autoencoder..........................
# Training Parameters
learning_rate = 0.01
num_steps = 30000

# Network Parameters
n_hidden_attribute = 500 # 1st layer num features
n_hidden_topology = 200 # 1st layer num features
n_hidden_encode = 100 # 2nd layer num features

# tf Graph input (only pictures)
X_attribute = tf.placeholder(tf.float32, [None, number_of_attribute])
X_topology = tf.placeholder(tf.float32, [None, number_of_nodes])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([number_of_attribute, n_hidden_attribute])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_attribute, n_hidden_encode])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_encode, n_hidden_attribute])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_attribute, number_of_attribute])),
}
biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_attribute])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_encode])),
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_attribute])),
    'decoder_b2': tf.Variable(tf.zeros([number_of_attribute])),
}

weights1 = {
    'encoder_h1': tf.Variable(tf.random_normal([number_of_nodes, n_hidden_topology])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_topology, n_hidden_encode])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_encode, n_hidden_topology])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_topology, number_of_nodes])),
}
biases1 = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_topology])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_encode])),
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_topology])),
    'decoder_b2': tf.Variable(tf.zeros([number_of_nodes])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with leaky_relu activation #1
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with leaky_relu activation #2
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with leaky_relu activation #1
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with leaky_relu activation #2
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

def encoder1(x):
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights1['encoder_h1']), biases1['encoder_b1']))
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights1['encoder_h2']), biases1['encoder_b2']))
    return layer_2

def decoder1(x):
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights1['decoder_h1']), biases1['decoder_b1']))
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights1['decoder_h2']), biases1['decoder_b2']))
    return layer_2

#object function.................................
def GetP(encoded1, encoded2):
    return tf.reciprocal(tf.add(tf.constant(1.0), tf.exp(tf.negative(tf.matmul(encoded1, tf.transpose(encoded2))))))

def GetFirstOrder(H):
    return tf.negative(tf.reduce_sum(tf.multiply(tf.log(H), matrix_topology)))

def GetMinJ(P):
    p = tf.add(P, tf.multiply(100.0, matrix_topology))
    return tf.one_hot(indices=tf.argmin(p, axis=1), depth=number_of_nodes, dtype=tf.float32)

def GetConsistent(H, P):
    former = tf.negative(tf.reduce_sum(tf.multiply(tf.log(H), Diagonal_matrix)))
    index = GetMinJ(P)
    later = tf.negative(tf.reduce_sum(tf.multiply(tf.log(tf.subtract(1.0, H)), index)))
    return tf.add(former, later)

def GetCost(encodedM, encodedZ):
    encodedM = tf.div(encodedM, 100)
    encodedZ = tf.div(encodedZ, 100)
    Pmm = GetP(encodedM, encodedM)
    Pzz = GetP(encodedZ, encodedZ)
    Pmz = GetP(encodedM, encodedZ)
    P = tf.matmul(encodedM, tf.transpose(encodedZ))

    first_order_cost_m = GetFirstOrder(Pmm)
    first_order_cost_z = GetFirstOrder(Pzz)
    
    first_order_cost = tf.add(first_order_cost_m, first_order_cost_z)

    return tf.add(first_order_cost, GetConsistent(Pmz, P))

# Construct model...................................
encoder_op = encoder(X_attribute)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X_attribute

encoder_op1 = encoder1(X_topology)
decoder_op1 = decoder1(encoder_op1)

y_pred1 = decoder_op1
y_true1 = X_topology

# Define loss and optimizer, minimize the squared error
loss = tf.add(tf.reduce_sum(tf.pow(y_true - y_pred, 2)), tf.reduce_sum(tf.pow(y_true1 - y_pred1, 2)))
loss = tf.add(loss, GetCost(encoder_op1, encoder_op))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, 30):
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X_attribute: matrix_attribute, X_topology: matrix_attribute_high_order})

        print('Step %i: Minibatch Loss: %f' % (i, l))
    encoder_result = sess.run(encoder_op, feed_dict={X_attribute: matrix_attribute})
    encoder_result1 = sess.run(encoder_op1, feed_dict={X_topology: matrix_attribute_high_order})
    OutputFile(encoder_result.tolist(), encoder_result1.tolist())