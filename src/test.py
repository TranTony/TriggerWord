#------------------------------------------------------------
# test.py
#
# Inference for Kaggle Tensorflow speech competition

from os import listdir
from os.path import join, isfile, isdir, basename, splitext
import sys
import argparse
import tensorflow as tf
import numpy as np
import time
from timeit import default_timer as timer
import yaml
import util
import models

targetWords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house', 'marvin', 'nine', 'one', 'seven', 'sheila', 'six', 'three', 'tree', 'two', 'wow', 'zero']
#targetWords = ['yes', 'no']

def datasetTestBuildDataset(audioPath,PARAMS):
    dataset = []
    testFiles = listdir(audioPath)
    with open('./data/train/audio/testing_list.txt') as f:
        tlist = [s.strip() for s in f.readlines()]

    if PARAMS['trainLimitInput']:
        testFiles = testFiles[:PARAMS['trainLimitInput']]

    testFiles_dir = [f for f in testFiles if isdir(join(audioPath, f))]
        
    for fdirname in testFiles_dir:
        fdirpath = join(audioPath, fdirname) 
        tfile = listdir(fdirpath)       
        for fname in tfile:
            fpath = join(fdirpath, fname)
            if not(fname.endswith('.wav') and isfile(fpath)):
                continue
            if not(util.which_set(fname, PARAMS['validationPercentage'], 10) == 'testing'):
                continue
            data, samprate = util.readWavFile(fpath)
            if len(data) < samprate:
                pad = np.zeros((samprate,1))
                start = (len(pad)-len(data))//2
                pad[start:start+len(data)] = data
                data = pad

            if len(data) > samprate:
                start = (len(data) - samprate)//2
                data = data[start:start+samprate]

            dataset.append({'file':fpath, 'label': fdirname, 'data':data, 'samprate': samprate})
                
    return dataset


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',type=str, default='./data/train/audio',
                        dest='audioDir',
                        help='Directory containing audio files')
    parser.add_argument('-c', type=str,
                        dest='chkpointFile',
                        required = True,
                        help='Checkpoint filename')
    parser.add_argument('-p', type=str, default='./params.yml',
                        dest='paramFile',
                        help='Parameter file')    
    parser.add_argument('-o', type=str,
                        dest='outputPath',
                        required = True,
                        help='Path to store submission file')
    FLAGS, unparsed = parser.parse_known_args()

    PARAMS = yaml.load(open(FLAGS.paramFile,'r'))
    
    print('Loading audio data...')
    audioPath = FLAGS.audioDir
    dataset = datasetTestBuildDataset(audioPath, PARAMS)

    labels = ['unknown','silence'] + targetWords
    noutputs = len(labels)
    
    # parse one audio file to get types and dimensions
    tmpfeatures = util.doMFCC(dataset[0]['data'], PARAMS)       
    nsteps  = tmpfeatures.shape[0]
    ninputs = tmpfeatures.shape[1]
    

    tf.reset_default_graph()
    isTraining = tf.placeholder(tf.bool, name='istraining')
    
    with tf.device("/gpu:0"): #should be gpu
        batch_data    = tf.placeholder(tf.float32, shape=[None,nsteps,ninputs], name='batch_data')
        batch_labels  = tf.placeholder(tf.int32, shape=[None], name='batch_labels')
        logits        = models.conv2DRnn(batch_data, noutputs, isTraining)                
        xentropy      = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        loss          = tf.reduce_mean(xentropy, name = "loss")
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        optimizer     = tf.train.AdamOptimizer(learning_rate = learning_rate)
        training_op   = optimizer.minimize(loss)
        class_probs   = tf.nn.softmax(logits)

        
    with tf.device("/cpu:0"):
        correct     = tf.nn.in_top_k(logits, batch_labels, 1)
        accuracy    = tf.reduce_mean(tf.cast(correct, tf.float32))
        prediction  = tf.argmax(class_probs,1, name = "prediction")
        
    # Restore model
    print('Restoring model..')
    chkpointFile = FLAGS.chkpointFile
    metaFile     = chkpointFile + '.meta'
    if not isfile(metaFile):
        print("Error, meta file doesnt exist {}".format(metaFile))
        sys.exit(1)

    test_results = []              
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,chkpointFile)

        print('Starting predictions...')
        batchCount = 0
        batchReportInterval = 10
        timeStart = timer()
        for batch in util.inputGenerator(dataset, False, None, PARAMS, 0):
            batchPredictions = sess.run(prediction, feed_dict={batch_data: batch['features'], isTraining: 0})
            names = [basename(n.decode('utf-8')) for n in batch['files']]
            #label = [e['labels'] for e in batch]
            label = [n.decode('utf-8') for n in batch['labels']]
            #label = []
            pred  = [labels[c] for c in batchPredictions]

            test_results.extend([(n,l,p) for n,l,p in zip(names, label, pred)])
            batchCount += 1
            print(batchCount)
            if (batchCount % batchReportInterval) == 0:
                timeEnd = timer()
                inferRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)                
                print("Batch {}  Rate {:.2f}".format(batchCount, inferRate))
                timeStart = timer()

    subFile =  splitext(basename(chkpointFile))[0] + "_sub_" + time.strftime('%Y%m%d_%H%M%S') + '.csv'
    of = open(join(FLAGS.outputPath, subFile),'w')
    of.write("fname,label,pred\n")
    for n,l,p in test_results:
        of.write("{},{},{}\n".format(n,l,p))
    of.close()
