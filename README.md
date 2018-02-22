# ck-request-asplos18-mobilenets-tvm
CK workflow for ReQuEST ASPLOS'18 submission: 
## Artifact check-list

## Installation

### Install global prerequisites

```
# sudo apt-get install libtinfo-dev 
```

### Install Collective Knowledge
```# pip install ck ```

### Install this CK repository with all dependencies (other CK repos to reuse artifacts)
```$ ck pull repo --url=https://github.com/ctuning/ck-request-asplos18-mobilenets-tvm-arm```

### Detect and test OpenCL driver
```$ ck detect platform.gpgpu --opencl ```

