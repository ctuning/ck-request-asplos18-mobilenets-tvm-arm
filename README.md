# Collective Knowledge workflow for image classification submitted to [ReQuEST at ASPLOS'18](http://cknowledge.org/request-cfp-asplos2018.html)

* **Title:** Optimizing Deep Learning Workloads on ARM GPU with TVM
* **Authors:** Lianmin Zheng, Tianqi Chen

## Artifact check-list (meta-information)

We use the standard [Artifact Description check-list](http://ctuning.org/ae/submission_extra.html) from systems conferences including CGO, PPoPP, PACT and SuperComputing.

* **Algorithm:** image classification
* **Program:** TVM/NNVM, ARM Compute Library, MXNet, OpenBLAS
* **Compilation:** g++
* **Transformations:**
* **Binary:** will be compiled on a target platform
* **Data set:** ImageNet 2012 validation (50,000 images)
* **Run-time environment:** Linux with OpenCL
* **Hardware:** Firefly-RK3399 with ARM Mali-T860MP4 or other boards with ARM Mali GPUs
* **Run-time state:** set by our scripts
* **Execution:** inference speed
* **Metrics:** total execution time; top1/top5 accuracy over some (all) images from the data set
* **Output:** classification result; execution time; accuracy
* **Experiments:** CK command line
* **How much disk space required (approximately)?** ~4GB
* **How much time is needed to prepare workflow (approximately)?** several hours (mainly native compilation of packages)
* **How much time is needed to complete experiments (approximately)?** from several hours for performance validation to several days for ImageNet accuracy validation (50000 images)
* **Collective Knowledge workflow framework used?** Yes
* **Original artifact:** https://github.com/merrymercy/tvm-mali
* **Publicly available?:** Yes
* **Experimental results:** https://github.com/ctuning/ck-request-asplos18-results-mobilenets-tvm-arm

## Installation 

**NB:** The `#` sign means `sudo`.

### Install global prerequisites (Ubuntu)

```
# sudo apt-get install libtinfo-dev 
```

```
# pip install numpy scipy decorator matplotlib
or
# pip3 install numpy scipy decorator matplotlib
```


### Install Collective Knowledge
```
# pip install ck
```

### Install this CK repository with all dependencies (other CK repos to reuse artifacts)
```
$ ck pull repo:ck-request-asplos18-mobilenets-tvm-arm
```

### Detect and test OpenCL driver
```
$ ck detect platform.gpgpu --opencl 
```


### Install libBLAS
```
$ sudo apt-get install libblas*
```

* To detect and register in ck :
```
ck detect soft:lib.blas
```

* To check the environment:
```
$ ck show env --tags=blas,no-openblas
```

A possible output:

``
cfe1e23a4472bb1d   linux-32    32 BLAS library api-3    32bits,blas,blas,cblas,host-os-linux-32,lib,no-openblas,target-os-linux-32,v0,v0.3
``

### Install OpenBLAS

```
$ ck install package:lib-openblas-0.2.18-universal
```

If you want to test other openblas version:

```
$ ck list package:lib-openblas* 
```


### Install LaPack

```
$ ck install package:lib-lapack-3.4.2
```

### Install or detect llvm/clang compiler v4+

```
$ ck install package --tags=compiler,llvm
```

Though above is the suggested method, you can also install **llvm** via apt and the detect it via CK.

```
# apt-get install llvm-4.0 clang-4.0
$ ck detect soft:compiler.llvm 
```

## Packages installation

### ARM Compute Library

```
$ ck install package:lib-armcl-opencl-17.12  --env.USE_GRAPH=ON --env.USE_NEON=ON --env.USE_EMBEDDED_KERNELS=ON 
```

To check/install other versions available via CK 

```
$ ck list package:lib-armcl-opencl-* 
$ ck install package --tags=lib,armcl env.USE_GRAPH=ON --env.USE_NEON=ON --env.USE_EMBEDDED_KERNELS=ON 
```

### MXNet with OpenBLAS

```
$ ck install package:lib-mxnet-master-cpu --env.USE_F16C=0
```

### NNVM / TVM 

```
$ ck install package:lib-nnvm-tvm-master-opencl 
```

## Original benchmarking (no real classification)

### ARM Compute Library client (OpenCL)
This program must be first compiled

```
$ ck compile program:request-armcl-inference 
```

and then executed as following:
```
$ ck run program:request-armcl-inference --cmd_key=all
```

You can also use "ck benchmark" command to automatically set CPU/GPU frequency to max, 
compile program, run it N times and perform statistical analysis on empirical characteristics:

```
$ ck benchmark program:request-armcl-inference --cmd_key=all
```

We validated results from the [authors](https://github.com/merrymercy/tvm-mali):

```
backend: ARMComputeLib-mali	model: vgg16	conv_method: gemm	dtype: float32	cost: 1.6511
backend: ARMComputeLib-mali	model: vgg16	conv_method: gemm	dtype: float16	cost: 0.976307
backend: ARMComputeLib-mali	model: vgg16	conv_method: direct	dtype: float32	cost: 3.99093
backend: ARMComputeLib-mali	model: vgg16	conv_method: direct	dtype: float16	cost: 1.61435
backend: ARMComputeLib-mali	model: mobilenet	conv_method: gemm	dtype: float32	cost: 0.172009
backend: ARMComputeLib-mali	model: mobilenet	conv_method: direct	dtype: float32	cost: 0.174635
```

''(CK program [meta](https://github.com/ctuning/ck-request-asplos18-mobilenets-tvm-arm/blob/master/program/request-armcl-inference/.cm/meta.json))''

### MXNet with OpenBLAS client (CPU)

``` 
$ ck run program:request-mxnet-inference  --cmd_key=all
 or
$ ck benchmark program:request-mxnet-inference  --cmd_key=all

```

We validated results from the [authors](https://github.com/merrymercy/tvm-mali):

```
backend: MXNet+OpenBLAS	model: resnet18	dtype: float32	cost:0.4145
backend: MXNet+OpenBLAS	model: mobilenet	dtype: float32	cost:0.3408
backend: MXNet+OpenBLAS	model: vgg16	dtype: float32	cost:3.1244
```

### NNVM/TVM client (OpenCL)

```
$ ck run program:request-tvm-nnvm-inference  --cmd_key=all 
 or
$ ck benchmark program:request-tvm-nnvm-inference  --cmd_key=all 

```

We validated results from the [authors](https://github.com/merrymercy/tvm-mali):

```
backend: TVM-mali	model: vgg16	dtype: float32	cost:0.9599
backend: TVM-mali	model: vgg16	dtype: float16	cost:0.5688
backend: TVM-mali	model: resnet18	dtype: float32	cost:0.1748
backend: TVM-mali	model: resnet18	dtype: float16	cost:0.1122
backend: TVM-mali	model: mobilenet	dtype: float32	cost:0.0814
backend: TVM-mali	model: mobilenet	dtype: float16	cost:0.0525
```


## Real classification (time and accuracy)

Original benchmarking clients did not include real classification in this ReQuEST submission. 
We therefore provided code for real image classification for each of the above CK programs.

### MXNet with OpenBLAS client (CPU)

``` 
$ ck run program:request-mxnet-inference  --cmd_key=classify
```

### NNVM/TVM client (OpenCL)


```
$ ck run program:request-tvm-nnvm-inference  --cmd_key=classify 
```

### ARM Compute Library client (OpenCL)





## Run Programs with real classification 
