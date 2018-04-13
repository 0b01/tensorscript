import tensorflow as tf
length = 100
arr = tf.Variable([str(i) for i in range(1, length+1)])
graph = tf.while_loop(
    lambda i, _: tf.less(i, length+1), 
    lambda i, _: (tf.add(i,1), tf.cond(
        tf.logical_and(tf.equal(tf.mod(i, 3), 0), tf.equal(tf.mod(i, 5), 0)),
        (lambda : tf.assign(arr[(i - 1)], 'FizzBuzz')),
        (lambda : tf.cond(tf.equal(tf.mod(i, 3), 0),
            (lambda : tf.assign(arr[(i - 1)], 'Fizz')),
            (lambda : tf.cond(tf.equal(tf.mod(i, 5), 0),
                (lambda : tf.assign(arr[(i - 1)], 'Buzz')),
                (lambda : arr))))))),
    [1, arr])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    idx, array = sess.run(graph)
    print array
