import Config
import model
from model import generate_model, generate_attend_model, generate_brnn_model
from datetime import datetime
from utils import loadData, getMiniBatch, decodeCaptions, \
    getImage, writeCaption

import numpy as np
import os
import sys
import argparse

import tensorflow as tf

mode = 'train'
args = None
savedModelName = 'model1.0.ckpt'
dirName = "/gpfs/home/kkasarapu/project/coco/"

def runModel(sess, data, op, model, keepProb=0.0, attend=False):
    captions, features, urls = getMiniBatch( data,
                                             batch_size=Config.batch_size,
                                             split='train')
    inp = captions[:, :-1]
    out = captions[:, 1:]
    loss = None

    if attend:
        _, loss = sess.run([op, model['total_loss']], 
                           feed_dict={model['features']: features, 
                                      model['captions_in']: inp,
                                      model['captions_out']: out})
    else:
        mask = (out != Config._null)

        _, loss = sess.run([op, model['total_loss']], 
                           feed_dict={model['image_feature']: features, 
                                      model['input_seqs']: inp, 
                                      model['target_seqs']: out, 
                                      model['input_mask']: mask, 
                                      model['keep_prob']: keepProb})
    
    return loss

def runValidation(sess, data, model, keepProb=0.0, attend=False, brnn=False):

    captions, features, urls = getMiniBatch( data,
                                             batch_size=Config.batch_size,
                                             split='val')
    if attend:
        pred = sess.run( [ model['sampled_captions'] ], 
                         feed_dict= { model['features'] : features } )
        
        return captions, pred, urls
    else:
        inp = captions[:, 0].reshape(-1, 1)
        final_preds = []
        current_pred = inp
        mask = np.zeros((batch_size, Config.padded_length))
        mask[:, 0] = 1
        if brnn:
            state_fw = sess.run( model['initial_state_fw'], 
                                 feed_dict={model['image_feature']: features, 
                                            model['keep_prob']: keepProb } )
            state_bw = sess.run( model['initial_state_bw'], 
                                 feed_dict={model['image_feature']: features, 
                                 model['keep_prob']: keepProb } )
    
            # start to generate sentences
            for t in range(Config.padded_length):
                current_pred, state_fw, state_bw = \
                        sess.run( [model['preds'], 
                                   model['final_state_fw'], 
                                   model['final_state_bw'] ], 
                        feed_dict={ model['input_seqs']: current_pred, 
                                    model['initial_state_fw']: state_fw,
                                    model['initial_state_bw']: state_bw,
                                    model['input_mask']: mask, 
                                    model['keep_prob']: keepProb } )
        
                current_pred = current_pred.reshape(-1, 1)
                final_preds.append(current_pred)
            return final_preds, urls

        else:
            
            state = sess.run( model['initial_state'], 
                              feed_dict={model['image_feature']: features, 
                                         model['keep_prob']: keepProb } )
        
            # start to generate sentences
            for t in range(Config.padded_length):
                current_pred, state = \
                            sess.run( [model['preds'], 
                                       model['final_state']], 
                                feed_dict={ 
                                    model['input_seqs']: current_pred, 
                                    model['initial_state']: state, 
                                    model['input_mask']: mask, 
                                    model['keep_prob']: keepProb } )
        
                current_pred = current_pred.reshape(-1, 1)
                final_preds.append(current_pred)

            return final_preds, urls
    
def main(_):
    data = loadData(args.dataDir, attend=(args.attend=='True'))
    tf.reset_default_graph()

    graph = tf.Graph()
    with graph.as_default():
        if args.attend:
            with tf.variable_scope(tf.get_variable_scope()):
                model = generate_attend_model()
                tf.get_variable_scope().reuse_variables()
                val_model = validate(max_len=20)
        else:
            if args.brnn:
                model = generate_brnn_model(mode=mode)
            else:
                model = generate_model(mode=mode)

        decayFunc = None
        learning_rate = tf.constant(Config.initial_learning_rate)

        if Config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = \
                        (Config.num_examples_per_epoch / Config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              Config.num_epochs_per_decay)
            
            def _decayFunc(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=Config.learning_rate_decay_factor,
                    staircase=True)
            
            decayFunc = _decayFunc

        op = tf.contrib.layers.optimize_loss( loss=model['total_loss'],
                                        global_step=model['global_step'],
                                        learning_rate=learning_rate,
                                        optimizer=Config.optimizer,
                                        clip_gradients=Config.clip_gradients,
                                        learning_rate_decay_fn=decayFunc )
        
        # initialize all variables 
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            
            num_epochs = Config.total_num_epochs

            num_train = data['train_captions'].shape[0]
            iterations_per_epoch = max(num_train / Config.batch_size, 1)
            num_iterations = int(num_epochs * iterations_per_epoch)

            epoch = 0
            loss_history = []
            
            print("Total training iter: ", num_iterations)
            time_now = datetime.now()

            for t in range(num_iterations):
                
                total_loss_value = runModel(sess, data, op, 
                                            model, 
                                            Config.lstm_dropout_keep_prob,
                                            attend=(args.attend=='True')) 
                loss_history.append(total_loss_value)

                if t % 50 == 0:
                    print('(Iteration %d / %d) loss: %f, and time "\
                    "elapsed: %.2f minutes' % (
                        t + 1, num_iterations, 
                        float(loss_history[-1]), 
                        (datetime.now() - time_now).seconds/60.0))

                if (t+1) % 5000 == 0:
                    temp_dir = os.path.join(args.sample_dir, 
                                            'temp_dir_{}//'.format(t+1))
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    captions_deco = None
                    urls = None
                    if args.attend:
                        capt, gen_caps, urls = runValidation(sess, 
                                                             data, 
                                                             val_model,
                                                             attend=True)
                        captions_deco = decodeCaptions(gen_caps[0], 
                                                 data['idx_to_word'])
                    else:
                        captions_pred, urls = runValidation(sess, 
                                                    data, 
                                                    model, 1.0,
                                                    brnn=(args.brnn=='True')) 
                        captions_pred = [unpack.reshape(-1, 1) \
                                         for unpack in captions_pred]
                        captions_pred = np.concatenate(captions_pred, 1)
                    
                        captions_deco = decodeCaptions(captions_pred, 
                                                       data['idx_to_word'])

                    for j in range(len(captions_deco)):
                        img_name = os.path.join(temp_dir, 
                                                'image_{}.jpg'.format(j))
                        try:
                            img = getImage(urls[j])
                            writeCaption(img, img_name, captions_deco[j])
                        except:
                            continue
                        
            if not os.path.exists(args.savedSession_dir):
                os.makedirs(args.savedSession_dir)
            save_path = model['saver'].save(sess, 
                                            os.path.join(
                                                args.savedSession_dir, 
                                                savedModelName))
            print("done. Model saved at: ", os.path.join(
                args.savedSession_dir, savedModelName))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--savedSession_dir',
      type=str,
      default= dirName + 'savedSession/',
      help="Directory where your created model / session will be saved."
  )
  parser.add_argument(
      '--dataDir',
      type=str,
      default=dirName + 'annotations/',
      help='Directory where all your training and validation data can be found.'
  )
  parser.add_argument(
      '--attend',
      type=str,
      default='',
      help='If attention model needs to be run. '
  )
  parser.add_argument(
      '--brnn',
      type=str,
      default='',
      help='If brnn model needs to be run. '
  )
  parser.add_argument(
      '--sample_dir',
      type=str,
      default=dirName + 'progress_sample/',
      help='Directory where all intermediate samples will be saved.'
  )
  args, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
