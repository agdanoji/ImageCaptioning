import os, json
import numpy as np
import h5py

import urllib, os, tempfile
from scipy.misc import imread
import matplotlib
matplotlib.use('Agg')
from matplotlib import image
import matplotlib.pyplot as plt
import hickle


directory = "/gpfs/home/kkasarapu/show_and_attend/coco/annotations/"

def getImage(url):
  try:
    f = urllib.urlopen(url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(f.read())
    img = imread(fname)
    return img
  except:
    print('Error: ', url)
    
def writeCaption(image, name, caption):
  """
  Write caption onto an image 
  """
  plt.imshow(image)
  plt.axis("off")
  plt.title(caption)
  plt.savefig(name)
  plt.close()

def loadData( base_dir=directory, attend=False ):
    allData = {}
    
    # loading train&val captions, and train&val image index 
    f = h5py.File(os.path.join(base_dir, 'coco2014_captions.h5'), 'r')
    for key, value in f.items():
        allData[key] = np.asarray(value)
    f.close()

    if attend:
        allData['train_features'] = hickle.load(os.path.join(base_dir, 'train2014_10000.features.hkl'))
        allData['val_features'] = hickle.load(os.path.join(base_dir, 'val2014_1000.features.hkl'))
    else:
        allData['train_features'] = np.load(os.path.join(base_dir, 'train2014_v3_pool_3.npy'))
        allData['val_features'] = np.load(os.path.join(base_dir, 'val2014_v3_pool_3.npy'))

    f = open(os.path.join(base_dir, 'coco2014_vocab.json'), 'r')
    dict_data = json.load(f)
    for key, value in dict_data.items():
        allData[key] = value
    f.close()

    # convert string to int for the keys 
    allData['idx_to_word'] = {int(key):value for key, value in allData['idx_to_word'].items()}

    f = open(os.path.join(base_dir, 'train2014_urls.txt'), 'r')
    allData['train_urls'] = np.asarray([line.strip() for line in f])
    f.close()

    f = open(os.path.join(base_dir, 'val2014_urls.txt'), 'r')
    allData['val_urls'] = np.asarray([line.strip() for line in f])
    f.close()

    if attend:
        newData = {}
        image_idxs = range(0, 10000)
        train_mask = []
        for y in range(len(allData['train_image_idx'])):
            if allData['train_image_idx'][y] in image_idxs:
                train_mask.append(y)

        captions = allData['train_captions'][train_mask]
        urls = allData['train_urls'][image_idxs]
        image_idxs2 = allData['train_image_idx'][train_mask]

        newData['train_features'] = allData['train_features']
        newData['train_captions'] = captions
        newData['train_image_idx'] = image_idxs2
        newData['train_urls'] = urls


        val_image_idxs = range(0, 1000)
        val_mask = []
        for y in range(len(allData['val_image_idx'])):
            if allData['val_image_idx'][y] in val_image_idxs:
                val_mask.append(y)

        val_captions = allData['val_captions'][val_mask]
        val_urls = allData['val_urls'][val_image_idxs]
        val_image_idxs2 = allData['val_image_idx'][val_mask]
        
        newData['val_features'] = allData['val_features']
        newData['val_captions'] = val_captions
        newData['val_image_idx'] = val_image_idxs2
        newData['val_urls'] = val_urls

        newData[ 'idx_to_word' ] = allData['idx_to_word']
        newData[ 'word_to_idx' ] = allData['word_to_idx']

        return newData
    
    return allData

def getMiniBatch(data, batch_size=100, split='train'):
    splitSize = data[ '%s_captions' % split ].shape[0]
    mask = np.random.choice(splitSize, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idx' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls
    

def decodeCaptions( captions, idx_to_word ):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded
