import tensorflow as tf
from datetime import datetime 
import Config
from model import generate_model
from utils import loadData, getMiniBatch, decodeCaptions, getImage, writeCaption
import numpy as np
import scipy.misc

verbose = True
mode = 'inference'
directory = '/gpfs/home/kkasarapu/project/coco/'

def test_rnn(sess, data, batch_size, model, keepProb):
    captions, features, urls = getMiniBatch( data,
                                             batch_size=batch_size,
                                             split='val')
    inp = captions[:, 0].reshape(-1, 1)
    #print(inp)
    final_preds = []
    current_pred = inp
    mask = np.zeros((batch_size, Config.padded_length))
    mask[:, 0] = 1
    
    state = sess.run( model['initial_state'], 
                      feed_dict={model['image_feature']: features, 
                                 model['keep_prob']: keepProb } )
    
    # start to generate sentences
    for t in range(Config.padded_length):
        current_pred, state = sess.run( [model['preds'], model['final_state']], 
                                        feed_dict={ model['input_seqs']: current_pred, 
                                                    model['initial_state']: state, 
                                                    model['input_mask']: mask, 
                                                    model['keep_prob']: keepProb } )
        
        current_pred = current_pred.reshape(-1, 1)
        final_preds.append(current_pred)

    return captions, final_preds, urls
    
    
# load data 
data = loadData(base_dir = directory+"annotations" )

TOTAL_INFERENCE_STEP = 10000
BATCH_SIZE_INFERENCE = 32
hyp = open(directory+"hyp.txt", "a")
ref1 = open(directory+"ref1.txt", "a")

# Build the TensorFlow graph and train it
g = tf.Graph()
with g.as_default():
    # Build the model.
    model = generate_model(mode, inference_batch = BATCH_SIZE_INFERENCE)
    
    # run training 
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
    
        sess.run(init)
    
        model['saver'].restore(sess, directory + "savedSession/model1.0_checkpoint160000.ckpt")
          
        print("Model restured! Last step run: ", sess.run(model['global_step']))
        
        for i in range(TOTAL_INFERENCE_STEP):
            orig_pred, captions_pred, urls = test_rnn(sess, data, BATCH_SIZE_INFERENCE, model, 1.0) # the output is size (32, 16)
            captions_pred = [unpack.reshape(-1, 1) for unpack in captions_pred]
            captions_pred = np.concatenate(captions_pred, 1)            
            captions_deco = decodeCaptions(captions_pred, data['idx_to_word'])
            orig_deco =  decodeCaptions(orig_pred, data['idx_to_word'])
            for j in range( len(captions_deco) ):
                cap = ' '.join(captions_deco[j].split()[:-1])
                ref = ' '.join(orig_deco[j].split()[1:-1])
                hyp.write(str(cap) + "\n")
                ref1.write(str(ref) + "\n")
                
            # for j in range(len(captions_deco)):
            #     img_name = directory + 'test/' + 'image_' + str(j) + '.jpg'
            #     img = getImage(urls[j])
            #     writeCaption(img, img_name, captions_deco[j])


def test_attend(sess, data, model ):
    captions, features, urls = getMiniBatch( data,
                                             batch_size=Config.batch_size,
                                             split='val')
    pred = sess.run( [ model['sampled_captions'] ],
                     feed_dict= { model['features'] : features } )

    return captions, pred, urls


with g.as_default():
    # Build the model.                                                                                                                                        
    model = generate_model()
    tf.get_variable_scope().reuse_variables()
    val_model = validate(max_len=20)

    # run training                                                                                                                                            
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        model['saver'].restore(sess, directory + "savedSession/model1.0_checkpoint40000.ckpt")

        print("Model restured! Last step run: ", sess.run(model['global_step']))

        for i in range(TOTAL_INFERENCE_STEP):
            capt, gen_caps, urls = test_attend(sess, data, val_model )
            gt = decodeCaptions(capt, data['idx_to_word'])
            #print("Ground truths: ", gt)                                                                                                                     
            #print( gen_caps )                                                                                                                                
            decoded = decodeCaptions(gen_caps[0], data['idx_to_word'])
            #print("Generated caption: \n", decoded )                                                                                                         

            for j in range( len(decoded) ):
                cap = ' '.join(decoded[j].split()[:-1])
                ref = ' '.join(gt[j].split()[1:-1])
                hyp.write(str(cap) + "\n")
                ref1.write(str(ref) + "\n")

hyp.close()
ref1.close()
