



import time
import logging
from datetime import datetime
import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from utils.general_utils import get_minibatches
# from tqdm import tqdm
import os
import pickle

# from .evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn





class Encoder(object):
    def __init__(self, size, vocab_dim):
        print("building encoder")
        self.size = size
        self.vocab_dim = vocab_dim
#         self.vert_layers = vert_layers . not yet implemented

    def attention(self, trigger, sequence):
        '''params:
                trigger: [batch, 2*self.size] float32
                sequence:[batch,max_steps, 2*self.size] float32'''
        similarity_mat = tf.get_variable(name = 'similarity_mat', shape = [2*self.size, self.size])
        trans_trigger = tf.transpose(tf.matmul(similarity_mat, tf.transpose(trigger))) #2*self.size, batch_size
        trans_trigger = tf.expand_dims(trans_trigger, axis = 2) #batch,2*self.size, 1
        attention_vect = tf.matmul(sequence, trans_trigger) #batch, max_steps, 1
        attention_vect = tf.nn.softmax(attention_vect, dim = 1)
        attention_weighted_sequence = sequence*attention_vect
        
        return attention_weighted_sequence #batch_size, max_steps
    
    def affine(self, final_state):
        return tf.contrib.layers.fully_connected(final_state, self.size)

    def encode(self, inputs, masks, encoder_state_input = None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with tf.variable_scope("encoder") as scope_encoder:
            
            #compute sequence length
            sequence_length = tf.reduce_sum(masks, axis = 1) 
            #create a forward cell
            fw_cell = tf.contrib.rnn.GRUCell(self.size)

            #create a backward cell
            bw_cell = tf.contrib.rnn.GRUCell(self.size)

            #pass the cells to bilstm and create the bilstm
            output_, final_state_ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype = tf.float32,\
                                                                    sequence_length = sequence_length,\
                                                                    initial_state_fw=encoder_state_input,\
                                                                    initial_state_bw =encoder_state_input)
            output = tf.concat([output_[0], output_[1]], axis = 2)
            final_state_i = tf.concat([final_state_[0], final_state_[1]], axis = 1)
            
            
            return output, final_state_i #[batch, max_steps, 2*self.size], [batch, 2*self.size]


class Decoder(object):
    def __init__(self, output_size):
        print("building decoder")
        self.output_size = output_size
        
    def decode(self, knowledge_rep, masks):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.
        
        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        with tf.variable_scope("decoder"):
            sequence_length = tf.reduce_sum(masks, axis=1)
            cell = tf.contrib.rnn.GRUCell(self.output_size)
            outputs, _ = tf.nn.dynamic_rnn(cell, knowledge_rep, sequence_length = sequence_length, dtype = tf.float32)
            logits = tf.contrib.layers.fully_connected(outputs, 2, activation_fn=None)
        return logits #[batch, max_length,2]


class Config():
    def __init__(self, embed_dim, evaluation_size, optimizer, minibatch_size, learning_rate, max_grad_norm):
        print("building config")
        self.max_grad_norm = max_grad_norm
        self.embed_dim = embed_dim
        self.evaluation_size = evaluation_size
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
    
    

class QASystem(object):
    def __init__(self, encoder, decoder, pretrained_embeddings, config, train_flag = True):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        print("building QASystem")
        self.encoder = encoder
        self.decoder = decoder
        
        self.pretrained_embeddings = tf.convert_to_tensor(pretrained_embeddings, dtype = tf.float32)
        
        
        self.embed_dim = config.embed_dim
        self.evaluation_size = config.evaluation_size
        self.optimizer = config.optimizer
        self.minibatch_size = config.minibatch_size
        self.learning_rate = config.learning_rate
        self.max_grad_norm = config.max_grad_norm
        
        
        self.q_masks = tf.placeholder(tf.int32, shape = [None, None])
        self.c_masks = tf.placeholder(tf.int32, shape = [None, None])
        self.q = tf.placeholder(tf.int32, shape = [None, None]) #batch_size x question_length
        self.c = tf.placeholder(tf.int32, shape = [None, None]) #batch_size x max_length
        self.a = tf.placeholder(tf.int32, shape = [None, None]) #batch_size x max_length
        self.skew_mat = tf.placeholder(tf.float32, shape = [None, None]) #batch x max_length
        
        self.global_step = tf.Variable(0, trainable = False)
        # ==== assemble pieces ====
        with tf.variable_scope("qa"):
            self.c_length = tf.shape(self.c_masks)[1]
            self.q_length = tf.shape(self.q_masks)[1]
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.make_optimizer()
            self.saver = self.saver_prot()

        # ==== set up training/updating procedure ====


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        with tf.variable_scope("question"):
            encoder_q, final_state = self.encoder.encode(self.q_e, self.q_masks)
            encoder_q_final_state = self.encoder.affine(final_state)
        with tf.variable_scope("context"):   
            encoder_c, encoder_c_final_state = self.encoder.encode(self.c_e, self.c_masks, encoder_state_input = encoder_q_final_state)
        with tf.variable_scope("attention"):
            knowledge_rep = self.encoder.attention(encoder_q_final_state, encoder_c)
        self.logits = self.decoder.decode(knowledge_rep, self.c_masks) #[batch, max_length,2]
        


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        
        with vs.variable_scope("loss"):
            mask = tf.cast(self.c_masks, tf.bool)
            unmasked_c_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.a, logits = self.logits)*self.skew_mat
            self.loss = tf.reduce_sum(tf.boolean_mask(unmasked_c_e, mask))/tf.cast(tf.reduce_sum(self.c_masks), tf.float32)
            
    def make_optimizer(self):
        optimizer = get_optimizer(self.optimizer)
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step = self.global_step, decay_steps = 10000,
                                                  decay_rate = 0.96, staircase = True)
        _optimizer_op = optimizer(learning_rate)
        gradients, variables = zip(*_optimizer_op.compute_gradients(self.loss))
        clipped_gradients, self.global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        self.optimizer_op = _optimizer_op.apply_gradients(zip(gradients, variables))
        self.optimizer_op = _optimizer_op.minimize(self.loss, global_step = self.global_step)
            

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            self.q_e = tf.nn.embedding_lookup(self.pretrained_embeddings, self.q)
            self.c_e = tf.nn.embedding_lookup(self.pretrained_embeddings, self.c)
            


    def answer_indices(self, session, dataset):
        """
        Returns the indices of predictions at positions over the length of the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.feed_dict(dataset)
        answers = input_feed[self.a]
        indices = tf.argmax(self.logits, axis = 2)
        output_feed = [indices]
        ind_out  = session.run(output_feed, input_feed)
        return ind_out, answers

    def answer(self, session, dataset):

        indices = np.array(self.answer_indices(session, dataset))
#         indices = np.sort(indices)
        a_s = indices[:,0]
        a_e = indices[:,1]

        return a_s, a_e #batch_size, 2

    def validate(self, session, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        input_feed = self.feed_dict(valid_dataset)
        output_feed = [self.loss]
        valid_cost = session.run(output_feed, input_feed)
        
        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
                        here the signature is [vq,vc,va]
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        for dataset_s in get_minibatches(dataset, sample):
            data_pad, data_pad_masks = self.pad(dataset_s)
            q, c, a = data_pad
            q_masks, c_masks = data_pad_masks[:2]
            dataset_a = [[q, c, a], [q_masks, c_masks]]
            
            ind_out, answers = np.array(self.answer_indices(session, dataset_a)) #batch, max_length
                
            gold = np.array(answers) #batch, max_lengt
            pp = np.sum(ind_out)
            gp = np.sum(gold)
            correct_predictions = np.logical_and(ind_out, gold)
            exact_preds = ind_out == gold
            cp = np.sum(correct_predictions)
            p = 0.
            if pp != 0.:
                p = cp/pp
            r = cp/gp
            f1 = 0.
            if (p + r) != 0.:
                f1 = 2*p*r/(p+r)
            em = np.sum(np.all(exact_preds, axis = 1))/sample 
            if log:
                logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
            break
        return f1, em
    
    
    def pad(self, datalist):
        padded = []
        masks = []
        for data in datalist:
            batch_len = len(data)
            m_len = 0
            for i in data:
                iter_len = len(i)
                if iter_len > m_len:
                    m_len = iter_len
                padded_list = [(t+[0]*(m_len-len(t))) for t in data]
                masks_list = [[1 if t != 0 else 0 for t in j] for j in padded_list]
            padded.append(padded_list)
            masks.append(masks_list)
        return padded, masks
    
    def feed_dict(self, dataset_feed):
        '''dataset is a list of [[q,c,a],[q_masks, c_masks]] or [[q,c], [[q_masks],[c_masks]]'''
        input_feed = {}  
        input_feed[self.q], input_feed[self.c] = dataset_feed[0][0:2]
        input_feed[self.q_masks], input_feed[self.c_masks] = dataset_feed[1]
        if len(dataset_feed[0]) == 3:
            answer_index = np.array(dataset_feed[0][2])
            answers = np.zeros_like(dataset_feed[0][1])
            for i in range(answers.shape[0]):
                answers[i, answer_index[i,0]:answer_index[i,1] + 1] = 1
            sum_ans = np.sum(answers, axis = 1)[:, np.newaxis] #batch,1
            not_ans = answers == 0 #batch, max_len
            sum_not_ans = np.sum(not_ans, axis = 1)[:, np.newaxis] #batch,1
            weights_1 = sum_not_ans/sum_ans #batch,1
            loss_weights = not_ans + answers*weights_1
            input_feed[self.skew_mat] = loss_weights/np.linalg.norm(loss_weights, axis = 1, keepdims = True)
            input_feed[self.a] = answers
        return input_feed
        
    
    def run_epoch(self, dataset, sess):
        '''dataset is a list [q,c,a]'''
        
        n_minibatches = 0.
        total_loss = 0.
        
        for dataset_mini in get_minibatches(dataset, self.minibatch_size):
            dat_pad, dat_pad_masks = self.pad(dataset_mini)
            dat_pad_masks = dat_pad_masks[:-1]
            n_minibatches += 1
            dataset_i = [dat_pad, dat_pad_masks]
            feed_dict = self.feed_dict(dataset_i)
            
            output = [self.optimizer_op , self.loss, self.global_norm]
            _, loss, global_norm = sess.run(output, feed_dict)
            if not n_minibatches % 100:
                print("n_minibatch = {}".format(n_minibatches), "loss: {}".format(loss), "global_norm{}".format(global_norm))
            total_loss += loss  
        return total_loss/n_minibatches
    
    def saver_prot(self):
        return tf.train.Saver()

    def train(self, session, dataset, epochs, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function.
                        In this implimentation it is passed down as a list of train and val:
                        [[train_q, train_c, train_a], [val_q, val_c, val_a]]
                        
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training


        results_path = os.path.join(train_dir, "{:%Y%m%d_%H%M%S}".format(datetime.now()))
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
       
        dataset_train, dataset_val = dataset    
        dataset_train = [[[int(x) for x in t.split()] for t in j] for j in dataset_train]
        dataset_val = [[[int(x) for x in t.split()] for t in j] for j in dataset_val]
        
        best_score, _ = self.evaluate_answer(session, dataset_val, sample=self.evaluation_size, log=True)
        
        
        #define optimizer and rate annealing
        
        
#         self.evaluate_answer(session, dataset_train, epoch = 1, sample=10, log=True)
        
        
        for epoch in range(epochs):
            logging.info("Epoch %d out of %d", epoch + 1, epochs)
            logging.info("Best score so far: "+str(best_score))
            loss = self.run_epoch(dataset_train, session)
            f1, em = self.evaluate_answer(session, dataset_val, sample=self.evaluation_size, log=True)
            logging.info("loss: " + str(loss) + " f1: "+str(f1)+" em:"+str(em))
            if f1 > best_score:
                best_score = f1
                logging.info("New best score! Saving model in %s", results_path)
                self.saver.save(session, results_path)    
            print("")

        return best_score
    