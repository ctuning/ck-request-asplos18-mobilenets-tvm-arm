"""
Benchmark inference speed on ImageNet
Updated by Grigori Fursin to support real image classification
"""

import os
import time
import sys
import argparse

import numpy as np

import mxnet as mx

from PIL import Image

input_size = 224

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

# returns list of pairs (prob, class_index)
def get_top5(all_probs):
  probs_with_classes = []
  for class_index in range(len(all_probs)):
    prob = all_probs[class_index]
    probs_with_classes.append((prob, class_index))
  sorted_probs = sorted(probs_with_classes, key = lambda pair: pair[0], reverse=True)
  return sorted_probs[0:5]

######################################################################
# Convert Gluon Block to MXNet Symbol
# See https://github.com/apache/incubator-mxnet/issues/9374
######################################################################
def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
        auxs[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs

def run_case(dtype, image):
    # Check image
    import os
    import json
    import sys

    STAT_REPEAT=os.environ.get('STAT_REPEAT','')
    if STAT_REPEAT=='' or STAT_REPEAT==None:
       STAT_REPEAT=10
    STAT_REPEAT=int(STAT_REPEAT)

    batch_size=1

    # FGG: set model files via CK env
    CATEG_FILE = '../synset.txt'
    synset = eval(open(os.path.join(CATEG_FILE)).read())

    files=[]
    val={}

    if image!=None and image!='':
       files=[image]
    else:
       ipath=os.environ.get('CK_ENV_DATASET_IMAGENET_VAL','')
       if ipath=='':
          print ('Error: path to ImageNet dataset is not set!')
          exit(1)
       if not os.path.isdir(ipath):
          print ('Error: path to ImageNet dataset was not found!')
          exit(1)

       # get all files
       d=os.listdir(ipath)
       for x in d:
           x1=x.lower()
           if x1.startswith('ilsvrc2012_val_'):
              files.append(os.path.join(ipath,x))

       files=sorted(files)

       STAT_REPEAT=1

       # Get correct labels
       ival=os.environ.get('CK_CAFFE_IMAGENET_VAL_TXT','')
       fval=open(ival).read().split('\n')

       val={}
       for x in fval:
           x=x.strip()
           if x!='':
              y=x.split(' ')
              val[y[0]]=int(y[1])

    # FGG: set timers
    import time
    timers={}

    # Get first shape (expect that will be the same for all)
    dt=time.time()
    image = Image.open(os.path.join(files[0])).resize((224, 224))
    if image.mode!='RGB': image=image.convert('RGB')
    timers['execution_time_load_image']=time.time()-dt

    dt=time.time()
    img = transform_image(image)
    timers['execution_time_transform_image']=time.time()-dt

    # load model
    from mxnet.gluon.model_zoo.vision import get_model
    from mxnet.gluon.utils import download

    model_path=os.environ['CK_ENV_MODEL_MXNET']
    model_id=os.environ['MXNET_MODEL_ID']

    block = get_model(model_id, pretrained=True, root=model_path)

    sym, arg_params, aux_params = block2symbol(block)

    sym = mx.sym.SoftmaxOutput(data=sym, name='softmax')

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = ['softmax_label'])

    eval_iter = mx.io.NDArrayIter(img, np.array([0.0]), batch_size, shuffle=False)

    mod.bind(data_shapes = eval_iter.provide_data, label_shapes = eval_iter.provide_label)

    mod.set_params(arg_params, aux_params)

    total_images=0
    correct_images_top1=0
    correct_images_top5=0

    # Shuffle files and pre-read JSON with accuracy to continue aggregating it
    # otherwise if FPGA board hangs, we can continue checking random images ...

    import random
    random.shuffle(files)

    if len(files)>1 and os.path.isfile('aggregate-ck-timer.json'):
       x=json.load(open('aggregate-ck-timer.json'))

       if 'total_images' in x:
          total_images=x['total_images']
       if 'correct_images_top1' in x:
          correct_images_top1=x['correct_images_top1']
       if 'correct_images_top5' in x:
          correct_images_top5=x['correct_images_top5']

    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])

    dt1=time.time()
    for f in files:
        total_images+=1

        print ('===============================================================================')
        print ('Image '+str(total_images)+' of '+str(len(files))+' : '+f)

        image = Image.open(os.path.join(f)).resize((224, 224))
        if image.mode!='RGB': image=image.convert('RGB')
        img = transform_image(image)

        # set inputs
        eval_iter = mx.io.NDArrayIter(img, np.array([0.0]), batch_size, shuffle=False)

        # perform some warm up runs
        # print("warm up..")
#        mod.predict(eval_iter)

        # execute
        print ('')
        print ("run ("+str(STAT_REPEAT)+" statistical repetitions)")
        dt=time.time()
        for repeat in range(0, STAT_REPEAT):
            prob = mod.predict(eval_iter)
        tcost=(time.time()-dt)/STAT_REPEAT

        timers['execution_time_classify']=tcost

        top1 = np.argmax(prob.asnumpy())

        print ('')
        print('TVM prediction Top1:', top1, synset[top1])

        top5=[]
        atop5 = get_top5(prob.asnumpy()[0])

        print ('')
        print('TVM prediction Top5:')
        for q in atop5:
            x=q[1]
            y=synset[x]
            top5.append(x)
            print (x,y)

        # Check correctness if available
        if len(val)>0:
           top=val[os.path.basename(f)]

           correct_top1=False
           if top==top1:
              correct_top1=True
              correct_images_top1+=1

           print ('')
           if correct_top1:
              print ('Current prediction Top1: CORRECT')
           else:
              print ('Current prediction Top1: INCORRECT +('+str(top)+')')

           accuracy_top1=float(correct_images_top1)/float(total_images)
           print ('Current accuracy Top1:   '+('%.5f'%accuracy_top1))

           correct_top5=False
           if top in top5:
              correct_top5=True
              correct_images_top5+=1

           print ('')
           if correct_top5:
              print ('Current prediction Top5: CORRECT')
           else:
              print ('Current prediction Top5: INCORRECT +('+str(top)+')')

           accuracy_top5=float(correct_images_top5)/float(total_images)
           print ('Current accuracy Top5:   '+('%.5f'%accuracy_top5))

           print ('')
           print ('Total elapsed time: '+('%.1f'%(time.time()-dt1))+' sec.')

           timers['total_images']=total_images
           timers['correct_images_top1']=correct_images_top1
           timers['accuracy_top1']=accuracy_top1
           timers['correct_images_top5']=correct_images_top5
           timers['accuracy_top5']=accuracy_top5

        timers['execution_time']=tcost

        with open ('tmp-ck-timer.json', 'w') as ftimers:
             json.dump(timers, ftimers, indent=2)

        with open ('aggregate-ck-timer.json', 'w') as ftimers:
             json.dump(timers, ftimers, indent=2)

        sys.stdout.flush()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help="Path to JPEG image.", default=None)
    args = parser.parse_args()

    # set parameter
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)

    # load model
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    dtype='float32'

    run_case(dtype, args.image)
