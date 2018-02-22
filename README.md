# ck-request-asplos18-mobilenets-tvm
CK workflow for ReQuEST ASPLOS'18 submission: 
## Artifact check-list



## Installation 

### Install global prerequisites (Ubuntu)

```
# sudo apt-get install libtinfo-dev 
```

```
# pip install scipy
or
# pip3 install scipy
```


### Install Collective Knowledge
```
# pip install ck
```

### Install this CK repository with all dependencies (other CK repos to reuse artifacts)
```
$ ck pull repo --url=https://github.com/ctuning/ck-request-asplos18-mobilenets-tvm-arm
```

``` 
$ ck pull repo:ck-mxnet
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

### Install or detect llvm/clang compiler

```
$ ck install package:compiler-llvm-4.0.0-universal
```

On * Firefly RK-3399 * install **llvm** via apt and the detect it via CK.

```
$ sudo apt-get install llvm-4.0 clang-4.0
```

```
$ ck detect soft:compiler.llvm 
```

## Packages installation

### ARM Compute Library

```
$ ck install package:lib-armcl-opencl-17.12  --env.USE_GRAPH=ON --env.USE_NEON=ON --env.USE_EMBEDDED_KERNELS=ON 
```

To check other versions available via CK 

```
$ ck list package:lib-armcl-opencl-* 
```

### MXNet with OpenBLAS

```
$ ck install package:lib-mxnet-master-cpu 
```

### NNVM / TVM 

```
$ ck install package:lib-nnvm-tvm-master-opencl 
```

## Run Programs 


### ARM Compute Library client
This program must be first compiled

```
$ ck compile program:tvm-arm 
```

``` 
$ ck compile program:  --cmd_key=all
```

### MXNet with OpenBLAS client

``` 
$ ck run program:tvm-mxnet  --cmd_key=all
```

## NNVM/TVM client

```
$ ck run program:tvm-nnvm  --cmd_key=all 
```

## Other options 

``` 
ck run program: * program_name * --cmd_key=help 
```

## FAQ 
