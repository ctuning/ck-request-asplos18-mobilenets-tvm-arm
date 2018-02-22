echo "\nHOW TO USE mxnet_test.py\n\n"
echo "./mxnet_test.py --model=all [Perform all the experiment supported]"
echo "./mxnet_test.py --model=[mobilenet || resnet18 || vgg16]"
echo "\n"

echo "\nHOW TO USE mxnet_test.py\n\n via CK\n\n"
echo "ck run --cmd_key=all [Perform all the experiment supported]"
echo "ck run --cmd_key=run-net --env.CK_MXNET_MODEL=[mobilenet || resnet18 || vgg16] --env.OMP_NUM_THREADS=[0 to #num of cores]"
echo "\n"

