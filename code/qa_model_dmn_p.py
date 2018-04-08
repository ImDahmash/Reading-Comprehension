
import time
import logging
from datetime import datetime
import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from utils.general_utils import get_minibatches
import os
import pickle
import functools
import copy

logging.basicConfig(level=logging.INFO)

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn



class EpisodicMemory():
    def __init__(self, cell, layers, encoder_state_size, ahl_units):
        self.cell = cell
        self.layers_n = layers
        self.encoder_size = 2*encoder_state_size
        self.ahl_units = ahl_units
        
        
    def _attention(self, q, m, s):
        '''q: [batch_size, 1, 2*encoder_state_size]
        m: [batch_size, 1, 2*encoder_state_size]
        s: [batch_size, sent_max_length, 2*encoder_state_size]'''
        
        s_q_dot = s * q #batch, sent_max_length, 2*encoder_state_size
        s_m_dot = s * m #batch, sent_max_length, 2*encoder_state_size
        s_q_diff = tf.abs(s-q) #batch, sent_max_length, 2*encoder_state_size
        s_m_diff = tf.abs(s-m) #batch, sent_max_length, 2*encoder_state_size
        z = tf.concat([s_q_dot, s_m_dot, s_q_diff, s_m_diff], axis = 2) #batch,sent_max_length,8*encoder_state_size
        z = tf.contrib.layers.fully_connected(tf.contrib.layers.fully_connected(z, self.ahl_units , activation_fn=tf.nn.tanh), 1, activation_fn = None)
        #batch,sent_max_length,1
        gates = tf.nn.softmax(z, dim = 1)
        return gates #batch, sent_max_length
        
        
    def _states(self,sent_context, seq_length):
            '''here the sent_context would be [batch,sent_max_length,encoder_state_size + 1]. 
            +1 for having appended global attention gate'''
            
            episode_states, m = tf.nn.dynamic_rnn(self.cell, sent_context, seq_length, dtype = tf.float32, parallel_iterations = 256)
            return tf.expand_dims(m, axis = 1) #batch, 1, 2*encoder_state_size
        #currently episode states wont be needed
        
    def knowledge_rep(self, sent_context, question, seq_length):
        '''sent_context: [batch, sent_max_length, 2*encoder_state_size]
        question: [batch, 2*encoder_state_size]
        seq_length: [batch_size] this is going to be the seq lengths for context representation at the sentence level'''
        question = tf.expand_dims(question, 1) #[batch_size, 1, 2*encoder_state_size]
        prev_m = question #[batch_size, 1, 2*encoder_state_size]
        for i in range(self.layers_n):
            with tf.variable_scope(str(i)) as scope:
                gates = self._attention(question, prev_m, sent_context) #[batch, sent_max_length]
                sent_context_g = tf.concat([sent_context, gates], axis = 2) #[batch,sent_max_length,2*encoder_state_size + 1]
                c = self._states(sent_context_g, seq_length) #[batch, 1, 2*encoder_state_size]
                inform_m = tf.concat([prev_m, c, question], axis = 2) #batch, 1, 6*encoder_state_size
                prev_m = tf.contrib.layers.fully_connected(inform_m, self.encoder_size) #batch, 1, 2*encoder_state_Size
        return tf.squeeze(prev_m, axis = 1) #batch, 2*encoder_state_size
    
    
    
############################################

class EpisodicMemoryCell(tf.contrib.rnn.LayerRNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super().__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    temp = inputs_shape.as_list()
    temp[1] = temp[1] - 1
    inputs_shape = tf.TensorShape(temp)
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    g = inputs[:,-1:]
    inputs = inputs[:,:-1]
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    r = math_ops.sigmoid(gate_inputs)
#     r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = g * c + (1 - g) * state
#     new_h = u * state + (1 - u) * c
    return new_h, new_h
        
class Encoder(object): #this is the same encoder to be used for encodign question as well as running the bidirectional gru in context encoding
    def __init__(self, encoder_size, vocab_dim):
        print("building encoder")
        self.size = encoder_size
        self.vocab_dim = vocab_dim
    
    def affine(self, final_state):
        return tf.contrib.layers.fully_connected(final_state, 2*self.size)

    def encode(self, inputs, masks, context = True):
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
            sequence_lengths = tf.reduce_sum(masks, axis = 1) 
            #create a forward cell
            fw_cell = tf.contrib.rnn.GRUCell(self.size)

            #pass the cells to bilstm and create the bilstm
            if context:
                bw_cell = tf.contrib.rnn.GRUCell(self.size)
                output, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, \
                                                                      bw_cell, inputs, \
                                                                      sequence_length = sequence_lengths, \
                                                                      dtype = tf.float32, \
                                                                      parallel_iterations = 256)
                output = tf.concat([output[0], output[1]], axis = -1)
                final_state = tf.concat([final_state[0], final_state[1]], axis = -1)
            else:
                output, final_state = tf.nn.dynamic_rnn(fw_cell, inputs, \
                                                        sequence_length = sequence_lengths,\
                                                        dtype = tf.float32,\
                                                        parallel_iterations = 256)
            return output, final_state #[batch, max_steps, self.size], [batch, self.size] or 2*self.size if context


class Decoder(object):
    def __init__(self, decoder_size):
        print("building decoder")
        self.decoder_size = decoder_size
        
        
    def decode(self, episode_mem, q, decoder_lengths,  context_word):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span
        
        :params
        episode_mem: [batch, 2*encoder_size]
        sos: [embed_size]
        q: [batch, 2*encoder_size]
        decoder_lengths: [batch] here just a tensor containing only 2s
        context_word:[batch, context_len, 2*encoder_size] word level representation of context over which attentions will be computed. gru output of encoder on context
                        all context_word will contain sos as the first token
        :return:
        """
        self.B, self.context_len = tf.shape(context_word)[0], tf.shape(context_word)[1]
        _, _, self.encoder_out_dim = context_word.shape.as_list()
#         self.encoder_size = encoder_out_dim/2
        self.context_embed = context_word #batch, cont_length, 2*encoder_size
        self.decoder_lengths= decoder_lengths
        self.q = q #batch, 2*encoder_size   
        decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_size)
        self.initial_state = tf.contrib.layers.fully_connected(episode_mem, self.decoder_size) #batch, self.decoder_size
        sos_step_embedded = tf.truncated_normal(shape = [self.encoder_out_dim])
        pad_step_embedded = tf.truncated_normal(shape = [self.encoder_out_dim])
        self.sos_step_embedded = tf.tile(tf.expand_dims(sos_step_embedded, 0), [self.B, 1])
        self.pad_step_embedded = tf.tile(tf.expand_dims(pad_step_embedded, 0), [self.B, 1])
        decoder_outputs, decoder_final_state, loop_state = tf.nn.raw_rnn(decoder_cell, self.loop_fn, parallel_iterations = 256) 
        #[batch, decoder_length, self.decoder.size], [batch, self.decoder_size], [[batch, decoder_length],[batch]] 
            
        return decoder_outputs, decoder_final_state, loop_state #[batch, max_length,2]
    
    def loop_fn_initial(self):
        initial_elements_finished = (0 >= self.decoder_lengths)  # all False at the initial step
        initial_input = tf.concat([self.sos_step_embedded, self.q], axis = 1) #batch, 4*encoder
        initial_cell_state = self.initial_state #batch, self.decoder_size
        initial_cell_output = None
        initial_loop_state = (tf.TensorArray(dtype = tf.float32, size = 2), \
                              tf.TensorArray(dtype = tf.int32, size = 2))
                              # we pass logits and collect them at the end to avoid calculating them again
    
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)
    
    def loop_fn_transition(self, time, previous_output, previous_state, previous_loop_state):
        output_logits = []
        prediction = []                    
        def get_next_input():
            with tf.variable_scope('var_scope', reuse = tf.AUTO_REUSE):
                weight = tf.get_variable('W', shape = [self.decoder_size, self.encoder_out_dim], dtype = tf.float32)
                bias = tf.get_variable('b', shape = [self.encoder_out_dim], dtype = tf.float32)
            output_query = tf.add(tf.matmul(previous_output, weight), bias) #batch, 2*self.encoder_dim
            output_query = tf.expand_dims(output_query, axis = 1) #batch, 1, 2*self.encoder_dim
            nonlocal output_logits, prediction
            output_logits = tf.reduce_sum(output_query * self.context_embed, axis = 2) #batch, cont_length
            prediction = tf.argmax(output_logits, axis=1, output_type = tf.int32) #batch
            #currently this is the only way to index with slices and lists
            #indexing is to be done out of batch, context_length, 2*encoder_dim
            idx_flat = tf.range(0, self.B)*self.context_len + prediction #batch
            _next_input = tf.nn.embedding_lookup(tf.reshape(self.context_embed, [-1, self.encoder_out_dim]), idx_flat) #batch,encoder_out_dim
            next_input = tf.concat([ _next_input, self.q], axis = 1) #batch, encoder_out_dim + encoder_out_dim
            return next_input
        elements_finished = (time >= self.decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                      # defining if corresponding sequence has ended

        finished = tf.reduce_all(elements_finished) # -> boolean scalar
#         next_input = tf.cond(finished, lambda: tf.concat([self.pad_step_embedded, self.q], axis = 1), get_next_input)
# tf.cond is currently throwing invalidargument error. Resolution: unknown. mostly it is a bug
        next_input = get_next_input()
        state = previous_state
        output = previous_output #batch, decoder_size
        previous_loop_state = (previous_loop_state[0].write(time - 1, output_logits), \
                               previous_loop_state[1].write(time - 1, prediction))
    
        

        return (elements_finished, 
                next_input,
                state,
                output,
                previous_loop_state)
    
    
    def loop_fn(self, time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:    # time == 0
            assert previous_output is None and previous_state is None
            return self.loop_fn_initial()
        else:
            return self.loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
    


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
    def __init__(self, encoder, decoder, episodic_mem_module, pretrained_embeddings, config, train_flag = True):
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
        self.episodic_mem_module = episodic_mem_module
        
        self.pretrained_embeddings = pretrained_embeddings
        
        self.embed_dim = config.embed_dim
        self.evaluation_size = config.evaluation_size
        self.optimizer = config.optimizer
        self.minibatch_size = config.minibatch_size
        self.learning_rate = config.learning_rate
        self.max_grad_norm = config.max_grad_norm
                
        self.q_masks = tf.placeholder(tf.int32, shape = [None, None])
        self.find42 = tf.shape(self.q_masks)
        self.c_masks = tf.placeholder(tf.int32, shape = [None, None])
        self.c_words_masks = tf.placeholder(tf.int32, shape = [None, None])
        self.q = tf.placeholder(tf.float32, shape = [None, None, self.embed_dim]) #batch_size x question_length
        self.c = tf.placeholder(tf.float32, shape = [None, None, self.embed_dim]) #batch_size x sent_max_length
        self.c_words = tf.placeholder(tf.float32, shape = [None, None, self.embed_dim]) #batch, context_length
        self.a = tf.placeholder(tf.int32, shape = [None, 2]) #batch_size x 2
        self.labels_s = self.a[:,0]
        self.labels_e = self.a[:,1]
        self.d_lengths = tf.placeholder(tf.int32, shape = [None]) #batch
        
        self.global_step = tf.Variable(0, trainable = False)
        # ==== assemble pieces ====
        with tf.variable_scope("qa"):
            self.c_length = tf.shape(self.c_masks)[1]
            self.q_length = tf.shape(self.q_masks)[1]
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
            encoder_q, final_state = self.encoder.encode(self.q, self.q_masks, context = False)
            encoder_q_final_state = self.encoder.affine(final_state) #batch, 2*encoder_size
        with tf.variable_scope("context"):   
            encoder_c, encoder_c_final_state = self.encoder.encode(self.c, self.c_masks)
        with tf.variable_scope('context_words'):
            encoder_c_words, _ = self.encoder.encode(self.c_words, self.c_words_masks)
        with tf.variable_scope("episodic_memory"):
            seq_length = tf.reduce_sum(self.c_masks, axis = 1)
            episode_mem = self.episodic_mem_module.knowledge_rep(encoder_c, encoder_q_final_state, seq_length) #batch, 2*encoder_size
        with tf.variable_scope("decoder_call"):
            _, _, out = self.decoder.decode(episode_mem, encoder_q_final_state, \
                                              self.d_lengths, encoder_c_words)
            #[batch, decoder_length, self.decoder.size], [batch, self.decoder_size], [[batch, decoder_length],[batch]]
            #convert tensorarrays  to tensors
            logits = out[0].stack() #2, batch, max_time_words
            preds = out[1].stack() #2, batch
            self.logits_s = logits[0]
            self.logits_e = logits[1]
            self.prediction_s = preds[0]
            self.prediction_e = preds[1]
    
    
    def _loss(self,logits, labels):
        B, L = tf.shape(logits)[0], tf.shape(logits)[1]
        stable_logits = logits - tf.reduce_max(logits, axis = 1, keep_dims = True)
        exp_logits = tf.exp(stable_logits)
        exp_logits_masked = exp_logits*tf.cast(self.c_words_masks, dtype = tf.float32)
        sum_exp_logits = tf.reduce_sum(exp_logits_masked, axis = 1, keep_dims = True)
        probs = exp_logits_masked/sum_exp_logits
        idx = tf.range(0, B)*L + labels
        correct_probs = tf.nn.embedding_lookup(tf.reshape(probs, [-1]), idx)
        cross_entropy = -tf.reduce_mean(tf.log(correct_probs))
        return cross_entropy


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        
        with vs.variable_scope("loss"):
            
            self.cross_ent_s = self._loss(self.logits_s, self.labels_s)
            self.cross_ent_e = self._loss(self.logits_e, self.labels_e)
            self.loss = self.cross_ent_s + self.cross_ent_e
            
            
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
        
            


    def answer_indices(self, session, dataset):
        """
        Returns the indices of predictions at positions over the length of the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.feed_dict(dataset)
        indices_s = self.prediction_s
        indices_e = self.prediction_e
        indices = [indices_s, indices_e]
        output_feed = indices
        ind_out  = session.run(output_feed, input_feed)
        return ind_out

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
        for dataset_minix in get_minibatches(dataset, sample):
            dataset_feed = self.prepare_feed_input(dataset_minix)
            ind_out = self.answer_indices(session, dataset_feed) #batch, 2
            answers = dataset_minix[2]
            gold = np.array(answers) #batch, 2
            preds = np.array(ind_out).T #batch, 2
            gold_lengths = gold[:,1] - gold[:, 0] + 1 #batch
            preds_lengths = preds[:,1] - preds[:, 0] + 1 #batch
            correct_mask = preds[:, 1] >= preds[:, 0] #batch
            correct_end_indices = np.where(gold[:,1] < preds[:,1], gold[:,1], preds[:,1])
            correct_start_indices = np.where(gold[:,0] > preds[:,0], gold[:,0], preds[:,0])
            
            correct_lengths = np.clip(correct_end_indices - correct_start_indices + 1, 0, None)
            correct_lengths = correct_mask*correct_lengths
            preds_lengths = preds_lengths*correct_mask
            gl = np.sum(gold_lengths)
            pl = np.sum(preds_lengths)
            exact_preds = np.logical_and((correct_lengths == gold_lengths),(correct_lengths == preds_lengths))
            cl = np.sum(correct_lengths)
            p = 0.
            if pl != 0:
                p = cl/pl
            r = cl/gl
            f1 = 0.
            if (p + r) != 0.:
                f1 = 2*p*r/(p+r)
            em = np.sum(exact_preds)
            if log:
                logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
            break
        return f1, em
    
    
    def pad(self, datalist, pad):
        
        padded = []
        masks = []
        D = len(datalist[0][0][0])
        for data in datalist:
            batch_len = len(data)
            m_len = 0
            for i in data:
                iter_len = len(i)
                if iter_len > m_len:
                    m_len = iter_len
            padded_list = [(t+[pad]*(m_len-len(t))) for t in data]
            masks_list = [[1 if np.all(t, axis = 0) != 0 else 0 for t in j] for j in padded_list]
            padded.append(padded_list)
            masks.append(masks_list)
        return padded, masks
    
    def feed_dict(self, dataset_feed):
        '''dataset is a list of [[q,c,a, c(words level)],[q_masks, c_masks, c_masks(words level)]] or [[q,c], [[q_masks],[c_masks]]'''
        input_feed = {}  
        batch_size = len(dataset_feed[0][0])
        input_feed[self.q], input_feed[self.c] = dataset_feed[0][0:2]
        input_feed[self.c_words] = dataset_feed[0][3]
        input_feed[self.q_masks], input_feed[self.c_masks], input_feed[self.c_words_masks] = dataset_feed[1]
        input_feed[self.d_lengths] = np.ones(batch_size) + 1
        if len(dataset_feed[0]) == 4:
            input_feed[self.a] = dataset_feed[0][2]
        
        return input_feed
    
    
    def sentence_reader(self, S):
        '''takes in a nested list of sentences in context and spits out
        sentence representation array padded to the length of sentence with 
        maximum number of sentences'''
        """D: vocab dimensions
           S: sentence list. len = batch, len[0] = no of sentences in first context
           len[0][0] = no of words in first sentence of first context"""
        D = len(S[0][0][0])
        sent_repr = [[functools.reduce(lambda x,y: x+y, [np.array([((1 - (j + 1)/len(M)) - ((d+1)/D)*(1 - 2*(j+1)/len(M)))*q for d,q in enumerate(p)]) \
                                                         for j,p in enumerate(M)]) for M in t] for t in S ]
        return sent_repr #[[sent1, sent2, sent3],[sent1, sen2], .. ]
    
    def embed_lookup(self, dataset):
        '''dataset: [q, c, c_words]
        c is going to be a list of lists of lists'''
        q = dataset[0]
        c = dataset[1]
        c_words = dataset[2]
        q_embed = [[self.pretrained_embeddings[i] for i in j] for j in q]
        c_embed = [[[self.pretrained_embeddings[i] for i in j] for j in t] for t in c]
        c_words_embed = [[self.pretrained_embeddings[i] for i in j] for j in c_words]
        return [q_embed, c_embed, c_words_embed]
    
    def prepare_feed_input(self, dataset_minix):
        '''Takes in input from get_minibatch and prepares the dataset to be fed to feed_dict
        Args:
          Inputs:
                dataset_minix: [q, c, a, c(word level)]
          Return:
                dataset_i: [[q_embed, c_sentence_embed, a (just indices), c_words_embed], dat_pad_masks]'''
        
        dataset_mini = self.embed_lookup([dataset_minix[0], dataset_minix[1], dataset_minix[3]]) #q,c,c(words) embedded
        dataset_c = self.sentence_reader(dataset_mini[1])
        dataset_mini[1] = dataset_c
        pad_embed = self.pretrained_embeddings[0]
        pad_input = dataset_mini
        dat_pad, dat_pad_masks = self.pad(pad_input, pad_embed)
        dataset_i = [[dat_pad[0], dat_pad[1], dataset_minix[2], dat_pad[2]], dat_pad_masks]
        return dataset_i
        
    
    def run_epoch(self, dataset, sess):
        '''dataset is a list [q,c,a, c(word level)]'''
        
        n_minibatches = 1.
        total_loss = 0.
        
        for dataset_mini in get_minibatches(dataset, self.minibatch_size):
            dataset_feed = self.prepare_feed_input(dataset_mini)
            feed_dict = self.feed_dict(dataset_feed)
            
            output = [self.optimizer_op , self.loss, self.global_norm]
            _, loss, global_norm = sess.run(output, feed_dict)
            total_loss += loss
            
            n_minibatches += 1.
            if not n_minibatches % 100:
                avg_loss = total_loss/n_minibatches
                print("n_minibatch = {}".format(n_minibatches), "loss: {}".format(avg_loss), "global_norm{}".format(global_norm))
              
        return total_loss/n_minibatches
    
    def saver_prot(self):
        return tf.train.Saver()

    def train(self, session, dataset, epochs, period_location, train_dir):
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
       
        dataset_train_raw, dataset_val_raw = dataset
        c_train = copy.deepcopy(dataset_train_raw[1])
        c_val = copy.deepcopy(dataset_val_raw[1])
        c_train = [[[int(j) for j in k.split()] for k in t] for t in [i.split(str(period_location)) for i in c_train]] #[[[w1,w2], [w3,w4]],[]..]
        c_val = [[[int(j) for j in k.split()] for k in t] for t in [i.split(str(period_location)) for i in c_val]]
        
        dataset_train_qa = [[[int(x) for x in t.split()] for t in j] for j in dataset_train_raw]
        dataset_val_qa = [[[int(x) for x in t.split()] for t in j] for j in dataset_val_raw]
        dataset_train = [dataset_train_qa[0], c_train, dataset_train_qa[2], dataset_train_qa[1]]
        dataset_val = [dataset_val_qa[0], c_val, dataset_val_qa[2], dataset_val_qa[1]]
        
        best_score, _ = self.evaluate_answer(session, dataset_val, sample=256, log=True)
        
        
        #define optimizer and rate annealing
        
        
        for epoch in range(epochs):
            logging.info("Epoch %d out of %d", epoch + 1, epochs)
            logging.info("Best score so far: "+str(best_score))
            loss = self.run_epoch(dataset_train, session)
            f1, em = self.evaluate_answer(session, dataset_train, sample=256, log=True)
            logging.info("loss: " + str(loss) + " f1: "+str(f1)+" em:"+str(em))
            if f1 > best_score:
                best_score = f1
                logging.info("New best score! Saving model in %s", results_path)
                self.saver.save(session, results_path)    
            print("")

        return best_score
    