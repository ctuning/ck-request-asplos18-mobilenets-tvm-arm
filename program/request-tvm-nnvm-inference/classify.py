"""
Benchmark inference speed on ImageNet
Example (run on Firefly RK3399):
python mali_imagenet_bench.py --target-host 'llvm -target=aarch64-linux-gnu' --host 192.168.0.100 --port 9090 --model mobilenet
"""

import time
import argparse
import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util, rpc
from tvm.contrib import graph_runtime as runtime
from PIL import Image

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

def run_case(model, dtype, image):
    # Check image
    import os
    import json
    import sys

    STAT_REPEAT=os.environ.get('STAT_REPEAT','')
    if STAT_REPEAT=='' or STAT_REPEAT==None:
       STAT_REPEAT=10
    STAT_REPEAT=int(STAT_REPEAT)

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
    if model == 'vgg16':
        net, params = nnvm.testing.vgg.get_workload(num_layers=16,
            batch_size=1, image_shape=image_shape, dtype=dtype)
    elif model == 'resnet18':
#        net, params = nnvm.testing.resnet.get_workload(num_layers=18,
#            batch_size=1, image_shape=image_shape, dtype=dtype)
         from mxnet.gluon.model_zoo.vision import get_model
         from mxnet.gluon.utils import download

         block = get_model('resnet18_v1', pretrained=True)

         # We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
         net, params = nnvm.frontend.from_mxnet(block)
         # we want a probability so add a softmax operator
         net = nnvm.sym.softmax(net)
    elif model == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(
            batch_size=1, image_shape=image_shape, dtype=dtype)
    else:
        raise ValueError('no benchmark prepared for {}.'.format(model))

    # compile
    opt_level = 2 if dtype == 'float32' else 1
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.mali(), shape={"data": data_shape}, params=params,
            dtype=dtype, target_host=args.target_host)

    # upload model to remote device
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)

    if args.host is not None:
        remote = rpc.connect(args.host, args.port)
        remote.upload(lib_fname)

        ctx = remote.cl(0)
        rlib = remote.load_module('net.tar')
        rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
    else:
        ctx = tvm.cl(0)
        rlib = lib
        rparams = params

    # create graph runtime
    dt=time.time()
    module = runtime.create(graph, rlib, ctx)
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype(dtype)))
    module.set_input(**rparams)
    timers['execution_time_create_run_time_graph']=(time.time()-dt)

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

    dt1=time.time()
    for f in files:
        total_images+=1

        print ('===============================================================================')
        print ('Image '+str(total_images)+' of '+str(len(files))+' : '+f)

        image = Image.open(os.path.join(f)).resize((224, 224))
        if image.mode!='RGB': image=image.convert('RGB')
        img = transform_image(image)

        # set inputs
        module.set_input('data', tvm.nd.array(img.astype(dtype)))
        module.set_input(**rparams)

        # execute
        print ('')
        print ("run ("+str(STAT_REPEAT)+" statistical repetitions)")
        dt=time.time()
        timer = module.module.time_evaluator("run", ctx, number=STAT_REPEAT)
        timers['execution_time_classify']=(time.time()-dt)/STAT_REPEAT
        tcost = timer()

        # get outputs
        tvm_output = module.get_output(
            0, tvm.nd.empty((1000,), dtype, ctx))

        top1 = np.argmax(tvm_output.asnumpy())

        top5=[]
        atop5 = get_top5(tvm_output.asnumpy())

        print ('')
        print('TVM prediction Top1:', top1, synset[top1])

        print ('')
        print('TVM prediction Top5:')
        for q in atop5:
            x=q[1]
            y=synset[x]
            top5.append(x)
            print (x,y)

        print ('')
        print("Internal T-cost: %g" % tcost.mean)

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

        timers['execution_time_classify_internal']=tcost.mean
        timers['execution_time']=tcost.mean

        with open ('tmp-ck-timer.json', 'w') as ftimers:
             json.dump(timers, ftimers, indent=2)

        with open ('aggregate-ck-timer.json', 'w') as ftimers:
             json.dump(timers, ftimers, indent=2)

        sys.stdout.flush()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet18', 'mobilenet', 'all'],
#                        help="The model type.")
    parser.add_argument('--model', type=str, required=False, choices=['vgg16', 'resnet18', 'mobilenet', 'all'],
                        help="The model type.", default='resnet18')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'])
    parser.add_argument('--host', type=str, help="The host address of your arm device.", default=None)
    parser.add_argument('--port', type=int, help="The port number of your arm device", default=None)
    parser.add_argument('--target-host', type=str, help="The compilation target of host device.", default=None)
    parser.add_argument('--image', type=str, help="Path to JPEG image.", default=None)
    args = parser.parse_args()

    # set parameter
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)

    # load model
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    if args.model == 'all': # test all
        for model in ['vgg16', 'resnet18', 'mobilenet']:
            for dtype in ['float32', 'float16']:
                run_case(model, dtype, args.image)
                time.sleep(10)

    else:  # test single
        run_case(args.model, args.dtype, args.image)
