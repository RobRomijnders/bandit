
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf  #version alpha 1.0
tf.logging.set_verbosity(tf.logging.ERROR)

class Model():
    def __init__(self, cf):
        width = cf['width']
        height = cf['height']
        num_a = cf['num_a']
        num_actions = cf['num_actions']
        lr = cf['lr']

        self.X = tf.placeholder(tf.float32, [None,height, width],"input_image")
        X_in = tf.expand_dims(self.X, 3) #MNIST digits have 1 color channel implicitly. Tensorflow conv expects an explicit color channel
        self.y = tf.placeholder(tf.float32, [None,num_actions],"correct_labels")
        self.keep_prob = tf.placeholder("float",name='keep_prob')

        with tf.variable_scope("conv", reuse=False) as scope:
            W1 = tf.get_variable('W1' ,[5, 5, 1, 16])
            b1 = tf.get_variable('b1' ,[16,])
            a1 = tf.nn.conv2d(X_in, W1, [1, 2, 2, 1], padding='SAME') + b1
            u1 = tf.contrib.layers.batch_norm(a1)
            h1 = tf.nn.relu(tf.nn.dropout(u1,self.keep_prob))

            W2 = tf.get_variable('W2' ,[5, 5, 16, 10])
            b2 = tf.get_variable('b2' ,[10 ,])
            a2 = tf.nn.conv2d(h1, W2, [1, 1, 1, 1], padding='SAME') + b2
            h2 = tf.nn.relu(tf.nn.dropout(a2,self.keep_prob))

            W3 = tf.get_variable('W3', [5, 5, 10, 10])
            b3 = tf.get_variable('b3', [10, ])
            a3 = tf.nn.conv2d(h2, W3, [1, 1, 2, 1], padding='SAME') + b3
            h3 = tf.nn.relu(tf.nn.dropout(a3, self.keep_prob))

            #################### Print sizes
            print('CONV1: kernels 5x5 to 16 channels. Resulting feature in [%i, %i, 16]'%(height/2, width/2))
            print('CONV2: kernels 5x5 to 10 channels. Resulting feature in [%i, %i, 10]' % (height / 2, width / 2))
            print('CONV3: kernels 5x5 to 10 channels. Resulting feature in [%i, %i, 10]' % (height / 2, width / 4))
            print('FC1: 100 neurons')
            ####################

            L = int((width/4 )*int(height/2)*10)
            W4 = tf.get_variable('W4' ,[L ,20])
            b4 = tf.get_variable('b4' ,[20 ,])
            h4_flat = tf.reshape(h3 ,[-1 ,L])
            u4_flat = tf.contrib.layers.batch_norm(h4_flat)
            a4 = tf.nn.xw_plus_b(u4_flat, W4, b4)  # in [batch_size, num_a]
            h4 = tf.nn.relu(a4)

            W5 = tf.get_variable('W5', [20, num_actions])
            b5 = tf.get_variable('b5', [num_actions, ])
            a5 = tf.nn.xw_plus_b(h4,W5,b5)
            _,self.pred = tf.nn.top_k(a5,num_a)



        with tf.variable_scope('opt', reuse=False) as scope:
            # cost = -1.0*tf.reduce_sum(tf.multiply(self.y,self.pred),1)
            cost = tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits=a5)
            self.cost_batch = tf.reduce_mean(cost)
            global_step = tf.Variable(0, trainable=False)
            lr_decay = tf.train.exponential_decay(lr, global_step, 2000, 0.5, staircase=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_decay)

            self.train_step = optimizer.minimize(self.cost_batch, global_step = global_step)

            self.init_op = tf.global_variables_initializer()
            self.all_saver = tf.train.Saver()

