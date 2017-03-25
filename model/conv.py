
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf  #version alpha 1.0
tf.logging.set_verbosity(tf.logging.ERROR)

class Model():
    def __init__(self, cf):
        width = cf['width']
        num_a = cf['num_a']
        num_actions = cf['num_actions']
        lr = cf['lr']

        self.X = tf.placeholder(tf.float32, [None,width, width],"input_image")
        X_in = tf.expand_dims(self.X, 3) #MNIST digits have 1 color channel implicitly. Tensorflow conv expects an explicit color channel
        self.y = tf.placeholder(tf.float32, [None,num_actions],"correct_labels")
        self.keep_prob = tf.placeholder("float",name='keep_prob')

        with tf.variable_scope("conv", reuse=False) as scope:
            W1 = tf.get_variable('W1' ,[5, 5, 1, 16])
            b1 = tf.get_variable('b1' ,[16,])
            a1 = tf.nn.conv2d(X_in, W1, [1, 2, 2, 1], padding='SAME') + b1
            h1 = tf.nn.relu(tf.nn.dropout(a1,self.keep_prob))

            W2 = tf.get_variable('W2' ,[5, 5, 16, 24])
            b2 = tf.get_variable('b2' ,[24 ,])
            a2 = tf.nn.conv2d(h1, W2, [1, 2, 2, 1], padding='SAME') + b2
            h2 = tf.nn.relu(tf.nn.dropout(a2,self.keep_prob))

            L = int((width/4 ) ** 2 *24)
            W3 = tf.get_variable('W3' ,[L ,num_actions])
            b3 = tf.get_variable('b3' ,[num_actions ,])
            h2_flat = tf.reshape(h2 ,[-1 ,L])
            h3 = tf.nn.xw_plus_b(h2_flat, W3, b3)  # in [batch_size, num_a]
            self.pred = tf.nn.log_softmax(h3)

        with tf.variable_scope('opt', reuse=False) as scope:
            cost = tf.reduce_sum(tf.multiply(self.y,self.pred),1)
            self.cost_batch = tf.reduce_mean(cost)
            global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

            self.train_step = optimizer.minimize(self.cost_batch, global_step = global_step)

            self.init_op = tf.global_variables_initializer()
            self.all_saver = tf.train.Saver()

