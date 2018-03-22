#!/bin/bash
#
# MXNET
declare -a mxnet_models=("mobilenet" "resnet18" "vgg16")

# ARM
declare -a arm_backend=("cl" "neon")
declare -a arm_conv=("direct" "gemm")
declare -a arm_models=("mobilenet" "vgg16")
declare -a arm_dtype=("float16" "float32")

#TVM 
declare -a tvm_models=("mobilenet" "resnet18" "vgg16")
declare -a tvm_dtype=("float16" "float32")



echo "MXNET"
name="mxnet"
d="float32"
b="openblas"
c="gemm"
for m in  "${mxnet_models[@]}"
do
#    ck benchmark program:tvm-mxnet --cmd_key=run-net --env.CK_MXNET_MODEL=$m --record --record_repo=local
#    ck run program:tvm-mxnet --cmd_key=run-net --env.CK_MXNET_MODEL=$m
     echo $name-$b-$m-$c-$d
     ck benchmark program:tvm-mxnet --cmd_key=run-net --env.CK_MXNET_MODEL=$m --record --record_repo=local --record_uoa=ck-request-asplos18-tvm-$name-$b-$m-$c-$d --tags=request,request-asplos18,tvm,$name,$b,$m,$c,$d
done

echo "TVM"
name="tvm"
b="opencl"
c="tvm-mali"
for m in "${tvm_models[@]}"
do
    for d in "${tvm_dtype[@]}"
    do
        #ck run program:tvm-nnvm --cmd_key=run-net --env.CK_TVM_MODEL=$m  --env.CK_TVM_DTYPE=$d
        echo $name-$b-$m-$c-$d
 #       ck benchmark program:tvm-nnvm --cmd_key=run-net --env.CK_TVM_MODEL=$m  --env.CK_TVM_DTYPE=$d --record --record_repo=local --record_uoa=ck-request-asplos18-tvm-$name-$b-$m-$c-$d --tags=request,request-asplos18,tvm,$name,$b,$m,$c,$d
    done
done


echo "ARM"
name="armcl"
for b in "${arm_backend[@]}"
do
    for m in "${arm_models[@]}"
    do
        for c in "${arm_conv[@]}"
        do
            for d in "${arm_dtype[@]}"
            do
                  #ck run program:tvm-arm --cmd_key=run-net --env.CK_ACL_BACKEND=$b --env.CK_ACL_MODEL=$m --env.CK_ACL_CONV_METHOD=$c --env.CK_ACL_DTYPE=$d  
                  echo $name-$b-$m-$c-$d
                  # ck benchmark program:tvm-arm --cmd_key=run-net --env.CK_ACL_BACKEND=$b --env.CK_ACL_MODEL=$m --env.CK_ACL_CONV_METHOD=$c --env.CK_ACL_DTYPE=$d --record_uoa=ck-request-asplos18-tvm-$name-$b-$m-$c-$d --tags=request,request-asplos18,tvm,$name,$b,$m,$c,$d
            done
        done
    done
done

