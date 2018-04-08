# Reading-Comprehension

python 3.5.4
tensorflow 1.7.0

This contains three different models for reading comprehension task set up to train on SQuAD

qa_model.py (train using train.py) is is a crude benchmark  running 2 GRU based encoders on question and context (preconditioned on question) followed by an attention mechanism. attentional vector is used in a decoder that classififes entire context as answer or non-answer

qa_model_dmn_p.py (train using traincolab_dmn_p.py) is a dynamic memory network with context embedding as described in https://arxiv.org/pdf/1503.08895.pdf 4.1 Sentence Representation. The word vectors used are not treated as variables i.e. they are not updated

qa_model_dmn_wordvar.py (train using traincolab_dmn_wordvar.py) is a dynamic memory network+ (https://arxiv.org/pdf/1603.01417.pdf) with Sentence Representation as hidden states collected at sentence end of a bidirectional GRU over context. The word vectors fed in are [tf.constant(wordvec); tf.Variable(wordvec)] (; is concatenation) thus training one version of word vectors and keeping one fixed. Decoder is a pointer network over context

get_started.sh is to download the data and GloVe

qa_data.py preprocesses the data

qa_answer.py and evaluate.py are still implemented  as part of training script
