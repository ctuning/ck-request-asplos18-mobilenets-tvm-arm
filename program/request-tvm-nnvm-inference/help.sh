#mali_imagenet_bench.py --target-host 'llvm -target=aarch64-linux-gnu' --host 192.168.0.100 --port 9090
#parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet18', 'mobilenet', 'all'],
#                        help="The model type.")
#    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'])
#    parser.add_argument('--host', type=str, help="The host address of your arm device.", default=None)
#    parser.add_argument('--port', type=int, help="The port number of your arm device", default=None)
#    parser.add_argument('--target-host', type=str, help="The compilation target of host device.", default=None)

echo "\nHOW TO USE mali_imagenet_bench.py\n\n"
echo "python mali_imagenet_bench.py --model=all [Perform all the experiment supported]"
echo "python mali_imagenet_bench.py --model=[mobilenet || resnet18 || vgg16] --dtype=[float16 || float32]"
echo "\n"

echo "\nHOW TO USE mxnet_test.py\n\n via CK\n\n"
echo "ck run --cmd_key=all [Perform all the experiment supported]"
echo "ck run --cmd_key=run-net --env.CK_TVM_MODEL=[mobilenet || resnet18 || vgg16]"
echo "\n"

