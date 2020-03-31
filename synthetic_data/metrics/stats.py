import sys
import tensorflow as tf
import numpy as np
import pandas as pd

class Stats():
	""" 
	Calculates statistics for the model

	Parameters
	----------
		meta_file : string, required
			The meta file for the Tensorflow model.
	"""
	def __init__(self, meta_file):

		sess = tf.Session()
		self.d_op, self.d_inpt = __restore_disc(meta_file)

	def __restore_disc(self, meta):
	    """
	    Restor the discriminator object
	    """

	    saver = tf.train.import_meta_graph(meta)

	    # Restore values
	    prefix = meta[:meta.index('.meta')]
	    saver.restore(sess, prefix)

	    # Get graph
	    graph = tf.get_default_graph()

	    # Get disc ops
	    disc_op = graph.get_tensor_by_name('Discriminator.4/BiasAdd:0')
	    try:
	        data_inpt = graph.get_tensor_by_name('RealData:0')
	    except KeyError as ke:
	        data_inpt = graph.get_tensor_by_name('Placeholder:0')

	    return disc_op, data_inpt

	def disc_test(self, data_file, batch_size=100):
		"""
		Runs the discriminator on a dataset for testing the loss

		Parameters
		----------
			data_file : string, required
				The SDV file on which the discriminator will be tested.
			batch_size: int, required
				The batch size to be used with the default value 100.
		Outputs
		-------
			The loss statistics.
		"""

		data = pd.read_csv(data_file)

		def get_every_n(a, n):
		    for i in range(a.shape[0] // n):
		        yield a[n * i:n * (i + 1)]

		loss = np.array([])

		for batch in get_every_n(data, batch_size):
		    loss = np.concatenate((loss, np.squeeze(
		    		sess.run(self.disc_op, feed_dict={self.data_inpt:batch}))))

		missing = data.shape[0] % n

		# If not evenly divisible
		if missing:
		    batch = np.zeros((n, data.shape[1]))
		    batch[:missing] = data[data.shape[0] - missing:]
		    loss = np.concatenate((loss, 
		    					   np.squeeze(sess.run(self.disc_op, 
		    					   					   feed_dict={self.data_inpt: batch})[:missing])))

		# Verify
		assert len(loss) == data.shape[0]

		# Print loss statistics
		print("Discriminator test statistics: ")
		print("Mean of loss: {}".format(loss.mean()))
		print("Standard deviation of loss: {}".format(loss.std()))
		print("Minimum value of loss: {}".format(loss.min()))
		print("Maximum value of loss: {}".format(loss.max()))
		print("Minimum value of absolute values of loss: {}".format(np.abs(loss).min()))