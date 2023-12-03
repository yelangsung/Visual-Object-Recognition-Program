import tensorflow as tf
import tensorflow.python.platform
import numpy as np
import os.path
import sys
import re

class NodeLookup(object):
    def __init__(self, uid_lookup_path, label_lookup_path):
        self.node_lookup = self.load(uid_lookup_path, label_lookup_path)

    def load(self, uid_lookup_path, label_lookup_path):
        """Loads a human readable English name for each softmax node."""
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def create_graph(graph_def_path):
    with tf.gfile.FastGFile(graph_def_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def main(graph_def_path, mode_id_path, human_label_path, image_path, num_top_predictions):
    inference_result = ['-----Result------']

    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    create_graph(graph_def_path)

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        node_lookup = NodeLookup(mode_id_path, human_label_path)

        top_k = predictions.argsort()[-num_top_predictions:][::-1]

        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            inference_result.append('%s (score = %.5f)' % (human_string, score))
            print('%s (score = %.5f)' % (human_string, score))

    return inference_result

# Example usage:
graph_def_path = 'C:/User/Yerang/Lab/graph_def.pb'
mode_id_path = 'C:/User/Yerang/Lab/imagenet_synset_to_human_label_map.txt'
human_label_path = 'C:/User/Yerang/Lab/imagenet_2017_challenge_label_map_proto.pbtxt'
image_path = 'C:/User/Yerang/Lab/image.jpg'
num_top_predictions = 5

result = main(graph_def_path, mode_id_path, human_label_path, image_path, num_top_predictions)
print(result)
