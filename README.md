# ck-request-asplos18-mobilenets-tvm
CK workflow for ReQuEST ASPLOS'18 submission: 
## Artifact check-list

## Installation

### Install global prerequisites

```
# sudo apt-get install libtinfo-dev 
```


### Install Collective Knowledge
```$ pip install ck ```

### Detect and test OpenCL driver
``` $ck detect platform.gpgpu --opencl ```

### Pre-install CK dependencies
``ck pull repo --url=https://github.com/ctuning/ck-request-asplos18-mobilenets-tvm-arm.git``
