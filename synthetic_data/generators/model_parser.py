"""
Freeze, convert, and gnerate data from tensorflow model
"""
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from google.protobuf import text_format


class ModelParser():

    def __init__(self):

        if not os.path.exists('gen_data'):
            os.makedirs('gen_data')

    def freeze_graph(self, graph, ckpt, output, node="\"Generator.3_1/Sigmoid\""):
        """
        Freezes the tensorflow graph

        Parameters
        ----------
        graph: string, required
            The name of the graph file stored as .pbtxt
        checkpoint: string, required
            The checkpoint to be used to freeze the graph
        output: string, required
            The name of the file to be used for saving the graph. Expected extension .pb
        node: string, optional
            Name for the output node

        """
        self.__freeze(graph, ckpt, output, node)

    def __freeze(self, graph, ckpt, output, node):

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "gen_data/")

            output_graph = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    node
            )       
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output, "wb") as f:
            f.write(output_graph.SerializeToString())

        print("Model frozen")

    def convert_graph(self, frozen_graph_file, ouptut_file='wgan_freeze.pbtxt'):
        """
        Retrieve the unserialized graph_def

        Parameters
        ----------
        frozen_graph_file: string, required
            The complete path including the file name for the frozen graph.
        output_file: string, option
            The name of the file as which the converted graph will be saved as.
        """

        with tf.gfile.GFile(frozen_graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        tf.io.write_graph(graph, 'gen_data' ouptut_file)

    def generate_data(graph, columns_file, num_files=10):
        """
        This function allows generation of data using the Tensorflow graph

        Parameters
        ----------
        graph: string, required
            The path including the file name of the graph
        columns_file: string, required
            The path including the file name which includes all the columns
        num_files: int, optional
            The count of files to be generated.
        """

        columns = json.load(open(columns_file))
        f = open(graph)
        graph_def = tf.GraphDef()

        gd = text_format.Merge(f.read(), graph_def)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(gd)

            inpt_name = [
                o.name for o in graph.get_operations()
                if o.name.endswith('RandomNoise')
            ][0]
            inpt = graph.get_tensor_by_name(inpt_name + ':0')
            gen_name = [
                o.name for o in graph.get_operations()
                if o.name.endswith('Generator.3_1/Sigmoid')
            ][0]
            gen = graph.get_tensor_by_name(gen_name + ':0')

            size_files = gen.shape[0].value

            with tf.Session(graph=graph) as sess:
                for i in range(num_files):
                    gen_data = sess.run(
                        gen,
                        feed_dict={inpt: np.random.normal(size=(size_files, 100))})

                    gen_data = pd.DataFrame(gen_data, columns=columns)

                    gen_data.to_csv(f'gen_data/txt_data_{i+1}.csv', index=False)
