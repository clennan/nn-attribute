
import tensorflow as tf


def propagate(images, model):
    batch_size = len(images)

    with tf.Session() as sess:
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

        with tf.variable_scope('forward_pass'):
            logits_tf = model.forward(x)

        with tf.variable_scope('relevance_backward_pass'):
            one_hot_class = tf.one_hot(tf.argmax(logits_tf[0]), 1000)
            relevances_tf = model.lrp(one_hot_class)

        tf.global_variables_initializer().run()

        logits, relevances = sess.run(
            [logits_tf, relevances_tf], feed_dict={x: images},
            )

    return logits, relevances
