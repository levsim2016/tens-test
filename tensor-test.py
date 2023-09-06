import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# rank_0_tensor = tf.constant(4)
# print(rank_0_tensor)
# rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
# print(rank_1_tensor)
# rank_2_tensor = tf.constant([[1, 2],
#                              [3, 4],
#                              [5, 6]], dtype=tf.float16)
# print(rank_2_tensor)
# rank_3_tensor = tf.constant([
#   [[0, 1, 2, 3, 4],
#    [5, 6, 7, 8, 9]],
#   [[10, 11, 12, 13, 14],
#    [15, 16, 17, 18, 19]],
#   [[20, 21, 22, 23, 24],
#    [25, 26, 27, 28, 29]],])

# print(rank_3_tensor)

rank_4_tensor = tf.zeros([3, 2, 4, 5])
print(rank_4_tensor)

# a = tf.constant([[1, 2],
#                  [3, 4]])
# b = tf.constant([[1, 1],
#                  [1, 1]]) # Could have also said `tf.ones([2,2])`

# print(tf.add(a, b), "\n")
# print(tf.multiply(a, b), "\n")
# print(tf.matmul(a, b), "\n")

# c = tf.constant([[4.0, 5.0, 6.0, 9,0]])

# Find the largest value
# print(tf.reduce_max(c))
# Find the index of the largest value
# print(tf.argmax(c))
# Compute the softmax
# print(tf.nn.softmax(c))
