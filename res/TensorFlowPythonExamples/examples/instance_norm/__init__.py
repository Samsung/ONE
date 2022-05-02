import tensorflow as tf

tf.compat.v1.disable_eager_execution()

sess = tf.Session()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(3, 3), name="Hole")
norm_ = tf.contrib.layers.instance_norm(in_)

# we need to save checkpoint to freeze dropped model
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
saver.save(sess, './ckpt/instance_norm.ckpt')

# use below command to freeze this model after running tfpem.py
'''
freeze_graph --input_graph instance_norm.pbtxt \
--input_binary=false \
--input_checkpoint=./ckpt/instance_norm.ckpt \
--output_node_names=InstanceNorm/instancenorm/add_1 \
--output_graph instance_norm_fr.pbtxt
'''
