#First, we take our input data, and we need to send it to hidden layer 1.
#Thus, we weight the input data, and send it to layer 1, where it will undergo the activation function,
#so the neuron can decide whether or not to fire and output some data to either the output layer, or another hidden layer.
#We will have three hidden layers in this example, making this a Deep Neural Network.
#From the output we get, we will compare that output to the intended output.
#We will use a cost function (alternatively called a loss function), to determine how wrong we are.
#Finally, we will use an optimizer function, "Adam Optimizer" in this case, to minimize the cost (how wrong we are).
#The way cost is minimized is by tinkering with the weights, with the goal of hopefully lowering the cost.
#How quickly we want to lower the cost is determined by the learning rate.
#The lower the value for learning rate, the slower we will learn, and the more likely we'll get better results.
#The higher the learning rate, the quicker we will learn, giving us faster training times, but also may suffer on the results.
#The act of sending the data straight through our network means we're operating a feed forward neural network.
#The adjusting of weights backwards is our back propagation.
#We do this feeding forward and back propagation however many times we want. The cycle is called an epoch.
#We can pick any number we like for the number of epochs, but you would probably want to avoid too many, causing overfitment.
#After each epoch, we've hopefully further fine-tuned our weights, lowering our cost and improving accuracy.
#When we've finished all of the epochs, we can test using the testing set.




import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)#mnist has data small manageable datasize.

#n_nodes can be any number according to our use 500 or 1500

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])#784 is the width in this case.height is None.
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} #biases are called here incase the input is 0. formula= ( input * weights) + bias

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    #actual modelling begins here. The input is multiplied with weights..etc
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # this does the activation part.

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)#till here we were training data

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

