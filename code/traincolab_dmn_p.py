



import os
import json

import tensorflow as tf
import numpy as np

from qa_model_dmn_p import Encoder, QASystem, Decoder, Config, EpisodicMemoryCell, EpisodicMemory
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

class _FLAGS():
    def __init__(self):
        self.learning_rate = 0.0003
        self.max_gradient_norm = 5.0
        self.dropout = 0.15
        self.batch_size = 128
        self.epochs = 40
        self.state_size = 250
        self.output_size =  750
        self.embedding_size = 100
        self.data_dir = "data/squad"
        self.train_dir = "traindmn_p"
        self.load_train_dir = ""
        self.log_dir = "log"
        self.optimizer = "adam"
        self.print_every = 1
        self.keep = 0
        self.vocab_path = "data/squad/vocab.dat"
        self.embed_path = ""
        self.evaluation_size = 500
        self.n_layers = 5
        self.attention_hidden_units = 500
FLAGS = _FLAGS()

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def get_device_name():
    return tf.test.gpu_device_name()

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = 'dir_node1'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir) #forlinux
#         os.system('rmdir "%s"' % global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir, True)
    return global_train_dir

def get_pretrained_embeddings(embed_path):
    glove = np.load(embed_path)
    return glove['glove']


def main():
    
    print('Device in use {}'.format(get_device_name()))

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = None

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    pretrained_embeddings = get_pretrained_embeddings(embed_path)
    
    config = Config(FLAGS.embedding_size, FLAGS.evaluation_size, FLAGS.optimizer, FLAGS.batch_size, FLAGS.learning_rate, \
                   FLAGS.max_gradient_norm)
    cell = EpisodicMemoryCell(2*FLAGS.state_size)
    episodicmemory = EpisodicMemory(cell, FLAGS.n_layers, FLAGS.state_size, FLAGS.attention_hidden_units)
    encoder = Encoder(FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(decoder_size=FLAGS.output_size)
    
    qa = QASystem(encoder, decoder, episodicmemory, pretrained_embeddings, config)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    
    with open(vocab_path, encoding = 'utf8') as file1:
        f = file1.readlines()
        period_location = [i for i in range(len(f)) if f[i] == '.\n']

    print((vars(FLAGS)))

    with open("data/squad"+"/train.ids.question", encoding = 'utf8') as t_i_q, open("data/squad" + "/train.ids.context", encoding = 'utf8') as t_i_c,\
         open("data/squad" + "/train.span", encoding = 'utf8') as t_s, open("data/squad" + "/val.ids.question", encoding = 'utf8') as v_i_q,\
         open("data/squad" + "/val.ids.context", encoding = 'utf8') as v_i_c, open("data/squad" + "/val.span", encoding = 'utf8') as v_s:
                q = t_i_q.readlines()
                c = t_i_c.readlines()
                a = t_s.readlines()
                vq = v_i_q.readlines()
                vc = v_i_c.readlines()
                va = v_s.readlines()
    dataset = [[q,c,a],[vq,vc,va]]
                

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, FLAGS.epochs, period_location, save_train_dir)
#         qa.evaluate_answer(sess, dataset_val, FLAGS.evaluation_size, log=True)

if __name__ == "__main__":
    main()
#     tf.app.run()