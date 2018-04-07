#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FizzBuzz():
    """FizzBuzz"""
    length = 30
    def __init__(self):
        with tf.name_scope("fizzbuzz"):
            self.array = tf.Variable([str(i) for i in range(1, self.length+1)], dtype=tf.string, trainable=False)
            self.graph = tf.while_loop(self.cond, self.body, [1, self.array],
                                shape_invariants=[tf.TensorShape([]), tf.TensorShape(self.length)],
                                back_prop=False)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, _):
        return (tf.less(i, self.length+1))

    def body(self, i, _):
        flow = tf.cond(
            tf.logical_and(tf.equal(tf.mod(i, 3), 0), tf.equal(tf.mod(i, 5), 0)),
            lambda: tf.assign(self.array[i - 1], 'FizzBuzz'),
            lambda: tf.cond(tf.equal(tf.mod(i, 3), 0),
                    lambda: tf.assign(self.array[i - 1], 'Fizz'),
                    lambda: tf.cond(tf.equal(tf.mod(i, 5), 0),
                            lambda: tf.assign(self.array[i - 1], 'Buzz'),
                            lambda: self.array
            )
            )
        )
        return (tf.add(i, 1), flow)


if __name__ == '__main__':
    fizzbuzz = FizzBuzz()
    ix, array = fizzbuzz.run()
    print(array)

