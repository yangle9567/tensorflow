#coding:utf-8
#0导入模块，生成模拟数据集。
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[int(x0 + x1 < 1)] for (x0,x1) in X]
print("X:\n",X)
print("Y:\n",Y_)
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
loss_mse=tf.reduce_mean(tf.square(y-y_))

# 定义损失函数和反向传播方法
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
# train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
# train_step=tf.train.AdamOptimizer(0.001).minimize(loss_mse)
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
    print("\n")

    STEPS=3000
    for i in range(STEPS):
        start=(i*BATCH_SIZE) % 32
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i % 500 ==0:
            total_loss=sess.run(loss_mse,feed_dict={x:X,y_:Y_})
            print("After %d training step(s),loss_mse on all data is %g" % (i,total_loss))


    # 输出训练后的参数取值。
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

#，只搭建承载计算过程的
#计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了。
#√会话（Session）： 执行计算图中的节点运算
    print("w1:\n", w1)
    print("w2:\n", w2)

