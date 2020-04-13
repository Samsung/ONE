#
# Copyright (C) 2017 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 3, 2}") # input tensor 0
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 2, 3, 2}") # input tensor 1
i3 = Input("op3", "TENSOR_FLOAT32", "{1, 2, 3, 2}") # input tensor 2
axis0 = Int32Scalar("axis0", 3)
r = Output("result", "TENSOR_FLOAT32", "{1, 2, 3, 6}") # output
model = model.Operation("CONCATENATION", i1, i2, i3, axis0).To(r)

# Example 1.
input0 = {i1: [-0.03203143, -0.0334147 , -0.02527265,  0.04576106,  0.08869292,
                0.06428383, -0.06473722, -0.21933985, -0.05541003, -0.24157837,
               -0.16328812, -0.04581105],
          i2: [-0.0569439 , -0.15872048,  0.02965238, -0.12761882, -0.00185435,
               -0.03297619,  0.03581043, -0.12603407,  0.05999133,  0.00290503,
                0.1727029 ,  0.03342071],
          i3: [ 0.10992613,  0.09185287,  0.16433905, -0.00059073, -0.01480746,
                0.0135175 ,  0.07129054, -0.15095694, -0.04579685, -0.13260484,
               -0.10045543,  0.0647094 ]}
output0 = {r: [-0.03203143, -0.0334147 , -0.0569439 , -0.15872048,  0.10992613,
                0.09185287, -0.02527265,  0.04576106,  0.02965238, -0.12761882,
                0.16433905, -0.00059073,  0.08869292,  0.06428383, -0.00185435,
               -0.03297619, -0.01480746,  0.0135175 , -0.06473722, -0.21933985,
                0.03581043, -0.12603407,  0.07129054, -0.15095694, -0.05541003,
               -0.24157837,  0.05999133,  0.00290503, -0.04579685, -0.13260484,
               -0.16328812, -0.04581105,  0.1727029 ,  0.03342071, -0.10045543,
                0.0647094 ]}

# Instantiate an example
Example((input0, output0))


'''
# The above data was generated with the code below:

with tf.Session() as sess:

    t1 = tf.random_normal([1, 2, 3, 2], stddev=0.1, dtype=tf.float32)
    t2 = tf.random_normal([1, 2, 3, 2], stddev=0.1, dtype=tf.float32)
    t3 = tf.random_normal([1, 2, 3, 2], stddev=0.1, dtype=tf.float32)
    c1 = tf.concat([t1, t2, t3], axis=3)

    print(c1) # print shape
    print( sess.run([tf.reshape(t1, [12]),
                     tf.reshape(t2, [12]),
                     tf.reshape(t3, [12]),
                     tf.reshape(c1, [1*2*3*(2*3)])]))
'''
