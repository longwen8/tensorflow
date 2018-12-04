import tensorflow as tf
import numpy as np
import io
import urllib.request
from PIL import Image


tf.reset_default_graph() 
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

saver = tf.train.Saver(write_version=tf.train.SaverDef.V1) 


sess = tf.Session()

sess.run(init)
saver.restore(sess , 'model/model.ckpt')

path = 'https://github.com/longwen8/tensorflow/blob/master/test/mnist/4.png?raw=true'
f = urllib.request.urlopen(path)
b = io.BytesIO(f.read())
img = Image.open(b).convert('L')

flatten_img = np.reshape(img,784)
xx = np.array([1 - flatten_img])
y = sess.run(y,feed_dict = {x:xx})
print('识别结果:')
print(np.argmax(y[0]))


