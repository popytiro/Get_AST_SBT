import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)
from sklearn.model_selection import train_test_split
import os
import io
import numpy as np
import re
import unicodedata
import urllib3
import shutil
import zipfile
import itertools
import sys
start_tem = int(sys.argv[1])
end_tem = int(sys.argv[2])
tem_valid_data = sys.argv[3]
# from google.colab import drive
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(24)

def read_file(filename):
    path = os.getcwd()
    path = os.path.join(path, filename)
    file = io.open(path,encoding='UTF-8')
    lines = file.read()
    file.close()
    return lines

def preprocess_sentence(s):
    s = s.rstrip().strip()
    s = '<start> ' + s + ' <end>'
    return s

def create_dataset(filename, num_samples):
    path = os.getcwd()
    path = os.path.join(path, filename)
    file = io.open(path,encoding='UTF-8')
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_samples]]
    return zip(*word_pairs)

def tokenize(input):
   tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
   tokenizer.fit_on_texts(input)
   sequences = tokenizer.texts_to_sequences(input)
  # print(max_len(sequences))
   sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
   return  sequences, tokenizer

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)

X_text,Y_text  = create_dataset("train_test_52040_less_1000_sbt_method.dataset", num_samples=5000)
X , X_tokenizer = tokenize(X_text)
Y, Y_tokenizer = tokenize(Y_text)
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
Tx = max_len(X)
Ty = max_len(Y)
def max_len(tensor):
    return max(len(t) for t in tensor)

input_vocab_size = 54
output_vocab_size = 1650
#output_vocab_size = 3000
#
print("input_vocab_size : ", input_vocab_size)
print("output_vocab_size : " ,output_vocab_size)

BATCH_SIZE = 64
BUFFER_SIZE = 4000
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32  


dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, return_state=True )
    
#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,BATCH_SIZE*[Tx])
        self.rnn_cell =  self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler, output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,attention_layer_size=dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units)

optimizer = tf.keras.optimizers.Adam()

def loss_function(y_pred, y):
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

decoderNetwork.attention_mechanism.memory_initialized

def train_step(input_batch, output_batch,encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,  initial_state =encoder_initial_cell_state)
        decoder_input = output_batch[:,:-1] # ignore <end>
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] #ignore <start>
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,encoder_state=[a_tx, c_tx],Dtype=tf.float32)
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,sequence_length=BATCH_SIZE*[Ty-1])
        logits = outputs.rnn_output
        #Calculate loss
        loss = loss_function(logits, decoder_output)
    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)
    #grads_and_vars  List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

checkpointdir = 'model'
chkpoint_prefix = os.path.join(checkpointdir, "chkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)

checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoderNetwork = encoderNetwork, decoderNetwork = decoderNetwork)

try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
except:
    print("No checkpoint found at {}".format(checkpointdir))


#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
        return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]

encoder_initial_cell_state = initialize_initial_state()
for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
    break
checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))

decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
print(decoderNetwork.decoder_embedding.variables[0].shape)

[print(var) for var in tf.train.list_variables(checkpointdir) if re.match(r'.*decoder_embedding.*',var[0])]

decoder_embedding_matrix = tf.train.load_variable(checkpointdir, 'decoderNetwork/decoder_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE')
print(decoder_embedding_matrix.shape)

#TEM EDIT
beam_width = 3

file_x = open(tem_valid_data,'r').readlines()
file_out = open('output_file_'+str(start_tem)+'_'+str(end_tem)+'_.txt','w')
file_out.write('expected,predicted,beam\n')
counter = 0
for k in file_x[start_tem:end_tem]:
    input_raw = k.split('\t')[0].lower()
    input_lines = input_raw.split("\n")
    input_lines = [preprocess_sentence(line) for line in input_lines]
    input_sequences = [[X_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,maxlen=Tx, padding='post')
    inp = tf.convert_to_tensor(input_sequences)
    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),tf.zeros((inference_batch_size, rnn_units))]
    encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
    a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,initial_state =encoder_initial_cell_state)
    start_tokens = tf.fill([inference_batch_size],Y_tokenizer.word_index['<start>'])
    end_token = Y_tokenizer.word_index['<end>']
    decoder_input = tf.expand_dims([Y_tokenizer.word_index['<start>']]* inference_batch_size,1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)
    encoder_memory = tfa.seq2seq.tile_batch(a, beam_width)
    decoderNetwork.attention_mechanism.setup_memory(encoder_memory)
    decoder_initial_state = decoderNetwork.rnn_cell.get_initial_state(batch_size = inference_batch_size* beam_width,dtype = Dtype)
    encoder_state = tfa.seq2seq.tile_batch([a_tx, c_tx], multiplier=beam_width)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
    decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoderNetwork.rnn_cell,beam_width=beam_width,output_layer=decoderNetwork.dense_layer)
    maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)
    (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix, start_tokens = start_tokens,end_token=end_token,initial_state = decoder_initial_state)
    inputs = first_inputs
    state = first_state  
    predictions = np.empty((inference_batch_size, beam_width,0), dtype = np.int32)
    beam_scores =  np.empty((inference_batch_size, beam_width,0), dtype = np.float32)                           
    for j in range(maximum_iterations):
        beam_search_outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
        inputs = next_inputs
        state = next_state
        outputs = np.expand_dims(beam_search_outputs.predicted_ids,axis = -1)
        scores = np.expand_dims(beam_search_outputs.scores,axis = -1)
        predictions = np.append(predictions, outputs, axis = -1)
        beam_scores = np.append(beam_scores, scores, axis = -1)
    for i in range(len(predictions)):
        output_beams_per_sample = predictions[i,:,:]
        score_beams_per_sample = beam_scores[i,:,:]
        for beam, score in zip(output_beams_per_sample,score_beams_per_sample) :
            seq = list(itertools.takewhile( lambda index: index !=2, beam))
            score_indexes = np.arange(len(seq))
            beam_score = score[score_indexes].sum()
            pred_txt = " ".join( [Y_tokenizer.index_word[w] for w in seq])
            print('Expected: ',k.split('\t')[1][:-1].lower(),' Predicted: ',pred_txt,' Beam score: ',str(beam_score))
            str_xx = k.split('\t')[1][:-1].lower()+','+pred_txt+','+str(beam_score)+'\n'
            file_out.write(str_xx)
    print(counter)
    counter = counter + 1
