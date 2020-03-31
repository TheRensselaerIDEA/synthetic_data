import sys
import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.Session()


def restore_disc(meta):
    """restor the discriminator object"""
    saver = tf.train.import_meta_graph(meta)

    # restore values
    prefix = meta[:meta.index('.meta')]
    saver.restore(sess, prefix)

    # get graph
    graph = tf.get_default_graph()

    # get disc ops
    disc_op = graph.get_tensor_by_name('Discriminator.4/BiasAdd:0')
    try:
        data_inpt = graph.get_tensor_by_name('RealData:0')
    except KeyError as ke:
        data_inpt = graph.get_tensor_by_name('Placeholder:0')

    return disc_op, data_inpt


def disc_test(data, disc_op, data_inpt, n=100):
    """run the discriminator on a dataset for testing the loss"""

    def get_every_n(a, n=100):
        for i in range(a.shape[0] // n):
            yield a[n * i:n * (i + 1)]

    loss = np.array([])

    for batch in get_every_n(data, n=n):
        loss = np.concatenate(
            (loss, np.squeeze(sess.run(disc_op, feed_dict={data_inpt:
                                                           batch}))))

    missing = data.shape[0] % n
    # if not evenly divisible
    if missing:
        batch = np.zeros((n, data.shape[1]))
        batch[:missing] = data[data.shape[0] - missing:]
        loss = np.concatenate(
            (loss,
             np.squeeze(
                 sess.run(disc_op, feed_dict={data_inpt: batch})[:missing])))

    # verify
    assert len(loss) == data.shape[0]

    return loss.mean(), loss.std(), loss.min(), loss.max(), np.abs(loss).min()


if len(sys.argv) != 3:
    print('Usage: python disc_tester.py <meta_file> <data_file>')
    sys.exit()

d_op, d_inpt = restore_disc(sys.argv[1])

data = train = pd.read_csv(sys.argv[2])
print(disc_test(data, d_op, d_inpt))