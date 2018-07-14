import tensorflow as tf
import Config



def generate_model(mode):
    init = tf.random_uniform_initializer( minval=-Config.initializer_scale,
                                          maxval=Config.initializer_scale )
    inputSequence = tf.placeholder( tf.int32, [None, None],
                                    name='inputSequence' ) 
    outputSequence = tf.placeholder( tf.int32, [None, None],
                                     name='outputSequence' ) 
    keepProb = tf.placeholder(tf.float32, name='keepProb')
    inputMask = tf.placeholder(tf.int32, [None, None], name='inputMask')

    imageFeatures = tf.placeholder(tf.float32, [Config.batch_size, 
                                                Config.image_feature_size],
                                   name='imageFeatures') 

    seqEmbed = None
    globalStep = tf.Variable(initial_value=0,
                             name="global_step",
                             trainable=False,
                             collections=[tf.GraphKeys.GLOBAL_STEP,
                                          tf.GraphKeys.GLOBAL_VARIABLES])
    
    with tf.variable_scope("seqEmbed"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable( name="map",
                                         shape=[Config.vocab_size,
                                                Config.embedding_size],
                                         initializer=init )
        seqEmbed = tf.nn.embedding_lookup(embedding_map, inputSequence)
        
    lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=Config.num_lstm_units,
                                         state_is_tuple=True )
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper( lstm_cell,
                                               input_keep_prob=keepProb,
                                               output_keep_prob=keepProb )

    with tf.variable_scope("lstm", initializer=init) as lstm_scope:
        zero_state = lstm_cell.zero_state(
                batch_size=Config.batch_size, dtype=tf.float32)
            
        with tf.variable_scope('image_embeddings'):
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=imageFeatures,
                num_outputs=Config.embedding_size,
                activation_fn=None,
                weights_initializer=init,
                biases_initializer=None)

        _, initial_state = lstm_cell(image_embeddings, zero_state)
        
        lstm_scope.reuse_variables()

        sequence_length = tf.reduce_sum(inputMask, 1)
        lstm_outputs, final_state = \
                            tf.nn.dynamic_rnn(cell=lstm_cell,
                                              inputs=seqEmbed,
                                              sequence_length=sequence_length,
                                              initial_state=initial_state,
                                              dtype=tf.float32,
                                              scope=lstm_scope)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope('logits'):
        W = tf.get_variable('W', [lstm_cell.output_size, 
                                  Config.vocab_size], initializer=init)
        b = tf.get_variable('b', [Config.vocab_size], 
                            initializer=tf.constant_initializer(0.0))
        
        logits = tf.add(tf.matmul(lstm_outputs, W), b)

    preds = tf.argmax(tf.nn.softmax(logits), 1)
    
    targets = tf.reshape(outputSequence, [-1])
    
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    
    batch_loss = tf.reduce_sum(tf.multiply(losses, weights))/tf.reduce_sum(inputMask)
    
    tf.contrib.losses.add_loss(batch_loss)
    total_loss = tf.contrib.losses.get_total_loss()

    return dict(
        total_loss = total_loss, 
        global_step = globalStep, 
        image_feature = imageFeatures, 
        input_mask = inputMask, 
        target_seqs = outputSequence, 
        input_seqs = inputSequence, 
        final_state = final_state,
        initial_state = initial_state, 
        preds = preds, 
        keep_prob = keepProb, 
        saver = tf.train.Saver()
    )

"""
The above model is based on Google's implementation from tensorflow

"""
    
weight_init = tf.contrib.layers.xavier_initializer()
const_init = tf.constant_initializer(0.0)
init = tf.random_uniform_initializer( minval=-Config.initializer_scale,
                                      maxval=Config.initializer_scale )

def attention(features, features_proj, h, reuse=False):
    
    with tf.variable_scope('attention_layer', reuse=reuse):
        w = tf.get_variable('w', [Config.num_lstm_units, Config.attend_size], 
                            initializer=weight_init)
        b = tf.get_variable('b', [Config.attend_size], 
                            initializer=const_init)
        w_att = tf.get_variable('w_att', [Config.attend_size, 1], 
                                initializer=weight_init)
        
        h_att = tf.add( tf.nn.relu(features_proj, \
                           tf.expand_dims( tf.add( tf.matmul(h, w), 1) , b) ))
        out_att = tf.reshape(tf.matmul(tf.reshape(h_att, 
                                                  [-1, Config.attend_size]), 
                                       w_att), [-1, Config.attend_layers])
        # creating weights from the attention output
        alpha = tf.nn.softmax(out_att)
        # finally weights the features with the alphas and get the contexts
        context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, 
                                name='context')
        return context

def generate_attend_model():
    # create a features placeholder for image features
    features2 = tf.placeholder(tf.float32, [Config.batch_size, 
                                           Config.attend_layers, 
                                           Config.attend_size],
                              name='features')
    mask = tf.to_float(tf.not_equal(captions_out, Config._null))

    features = tf.contrib.layers.batch_norm( inputs=features2,
                                             decay=0.95,
                                             center=True,
                                             scale=True,
                                             is_training=True,
                                             updates_collections=None,
                                             scope='conv_featuresbatch_norm')
    captions_in = tf.placeholder( tf.int32, [Config.batch_size, None],
                                  name='captions_in' )
    captions_out = tf.placeholder( tf.int32, [Config.batch_size, None],
                                  name='captions_out' )
    globalStep = tf.Variable(initial_value=0,
                             name="global_step",
                             trainable=False,
                             collections=[tf.GraphKeys.GLOBAL_STEP,
                                          tf.GraphKeys.GLOBAL_VARIABLES])

    with tf.variable_scope('initial_lstm'):
        # create the initial lstm state using the image features mean in each diection
        features_mean = tf.reduce_mean(imageFeatures, 1)
        w_h = tf.get_variable('w_h', [Config.attend_size, 
                                      Config.num_lstm_units], 
                              initializer=weight_init)
        b_h = tf.get_variable('b_h', [Config.num_lstm_units], 
                              initializer=const_init)
        # hidden layer
        h = tf.nn.relu( tf.add( tf.matmul(features_mean, w_h) , b_h))
        
        w_c = tf.get_variable('w_c', [Config.attend_size, 
                                      Config.num_lstm_units], 
                              initializer=weight_init)
        b_c = tf.get_variable('b_c', [Config.num_lstm_units], 
                              initializer=const_init)
        # cell state
        c = tf.nn.tanh( tf.add( tf.matmul(features_mean, w_c), b_c))
        
    
    with tf.variable_scope("x"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable( name="map",
                                         shape=[Config.vocab_size,
                                                Config.embedding_size],
                                         initializer=init )
        x = tf.nn.embedding_lookup(embedding_map, inputSequence)
    
    with tf.variable_scope('project_features'):
        # project the image features
        # basically flatten them so  that they can be used in attention layer
        w = tf.get_variable('w', [Config.attend_size, Config.attend_size], 
                            initializer=weight_init)
        features_flat = tf.reshape(imageFeatures, [-1, Config.attend_size])
        features_proj = tf.matmul(features_flat, w)
        features_proj = tf.reshape(features_proj, 
                                   [-1, Config.attend_layers, 
                                    Config.attend_size])

    loss = 0.0
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=Config.num_lstm_units)

    for t in range(Config.padded_length):
        # get the context for this time step using attention layer
        context = attention(features, 
                            features_proj, 
                            h, reuse=(t!=0))
            
        with tf.variable_scope('lstm', reuse=(t!=0)):
            # append the context and the current word and feed it to lstm
            _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], 
                                                     context],1), 
                                  state=[c, h])

        # calculate the logits separately
        with tf.variable_scope('logits', reuse=reuse):
          w_h = tf.get_variable('w_h', [Config.num_lstm_units, 
                                      Config.embedding_size], 
                              initializer=weight_init)
          b_h = tf.get_variable('b_h', [Config.embedding_size], 
                              initializer=const_init)
          w_out = tf.get_variable('w_out', [Config.embedding_size, 
                                          Config.vocab_size], 
                                initializer=weight_init)
          b_out = tf.get_variable('b_out', [Config.vocab_size], 
                                initializer=const_init)
        
          h = tf.nn.dropout(h, 0.5)
          h_logits = tf.add(tf.matmul(h, w_h), b_h)

          logits = tf.add(tf.matmul(h_logits, w_out), b_out)

        # calculate loss from each time step
        loss += tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=captions_out[:, t],logits=logits)*mask[:, t] )

    # finally calculate batch_loss and return for optimizing
    total_loss = loss*tf.reduce_sum(mask)/tf.to_float(Config.batch_size)
        
    return dict(
        total_loss = total_loss,
        global_step = globalStep,
        features = features2,
        captions_in = captions_in,
        captions_out = captions_out,
        saver = tf.train.Saver()
        )

def generate_brnn_model(mode, inference_batch=None):
    init = tf.random_uniform_initializer( minval=-Config.initializer_scale,
                                          maxval=Config.initializer_scale )
    inputSequence = tf.placeholder( tf.int32, [None, None],
                                    name='inputSequence' ) 
    outputSequence = tf.placeholder( tf.int32, [None, None],
                                     name='outputSequence' )
    keepProb = tf.placeholder(tf.float32, name='keepProb')
    inputMask = tf.placeholder(tf.int32, [None, None], name='inputMask')

    imageFeatures = tf.placeholder(tf.float32, [Config.batch_size, 
                                                Config.image_feature_size],
                                   name='imageFeatures') 

    seqEmbed = None
    globalStep = tf.Variable(initial_value=0,
                             name="global_step",
                             trainable=False,
                             collections=[tf.GraphKeys.GLOBAL_STEP,
                                          tf.GraphKeys.GLOBAL_VARIABLES])
    
    with tf.variable_scope("seqEmbed"), tf.device("/cpu:0"):
        
        embedding_map = tf.get_variable( name="map",
                                         shape=[Config.vocab_size,
                                                Config.embedding_size],
                                             initializer=init )
        seqEmbed = tf.nn.embedding_lookup(embedding_map, inputSequence)

    cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=Config.num_lstm_units,
                                       state_is_tuple=True )
    cell_bw = tf.contrib.rnn.BasicLSTMCell( num_units=Config.num_lstm_units,
                                       state_is_tuple=True )
    cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw,
                                             input_keep_prob=keepProb,
                                             output_keep_prob=keepProb )
    cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw,
                                             input_keep_prob=keepProb,
                                             output_keep_prob=keepProb )

    with tf.variable_scope("lstm", initializer=init) as lstm_scope:
        if mode == 'train':
            zero_state_fw = cell_fw.zero_state(
                batch_size=Config.batch_size, dtype=tf.float32)
            zero_state_bw = cell_bw.zero_state(
                batch_size=Config.batch_size, dtype=tf.float32)
        elif mode == 'inference':
            zero_state_fw = cell_fw.zero_state(
                batch_size=inference_batch, dtype=tf.float32)
            zero_state_bw = cell_bw.zero_state(
                batch_size=inference_batch, dtype=tf.float32)
            
        with tf.variable_scope('image_embeddings'):
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=imageFeatures,
                num_outputs=Config.embedding_size,
                activation_fn=None,
                weights_initializer=init,
                biases_initializer=None)

        _, initial_state_fw = cell_fw(image_embeddings, zero_state_fw)
        _, initial_state_bw = cell_bw(image_embeddings, zero_state_bw)
        
        lstm_scope.reuse_variables()

        sequence_length = tf.reduce_sum(inputMask, 1)
        outputs, (final_state_fw, final_state_bw) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                            cell_bw,
                                            inputs=seqEmbed,
                                            sequence_length=sequence_length,
                                            initial_state_fw=initial_state_fw,
                                            initial_state_bw=initial_state_bw,
                                            dtype=tf.float32,
                                            scope=lstm_scope)
        lstm_outputs = tf.concat(outputs, 2)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, 2*cell_fw.output_size])
        
        
    with tf.variable_scope('logits'):
        W = tf.get_variable('W', [2*cell_fw.output_size, 
                                  Config.vocab_size], initializer=init)
        b = tf.get_variable('b', [Config.vocab_size], 
                            initializer=tf.constant_initializer(0.0))
        
        logits = tf.add(tf.matmul(lstm_outputs, W), b)
        
    softmax = tf.nn.softmax(logits)
    preds = tf.argmax(softmax, 1)
    
    targets = tf.reshape(outputSequence, [-1])
    weights = tf.to_float(tf.reshape(inputMask, [-1]))
    
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                        tf.reduce_sum(weights),
                        name="batch_loss")
    
    tf.contrib.losses.add_loss(batch_loss)
    total_loss = tf.contrib.losses.get_total_loss()

    return dict(
        total_loss = total_loss, 
        global_step = globalStep, 
        image_feature = imageFeatures, 
        input_mask = inputMask, 
        target_seqs = outputSequence, 
        input_seqs = inputSequence, 
        final_state_fw = final_state_fw,
        final_state_bw = final_state_bw,
        initial_state_fw = initial_state_fw,
        initial_state_bw = initial_state_bw,
        softmax = softmax,
        preds = preds, 
        keep_prob = keepProb, 
        saver = tf.train.Saver()
    )
    



