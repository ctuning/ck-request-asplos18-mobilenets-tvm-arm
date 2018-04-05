
# System-level network classification and benchmarking

To build this program, you need ArmCL compiled with Graph API:

```
$ ck pull all
$ ck install package:lib-armcl-opencl-17.12 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph --env.USE_EMBEDDED_KERNELS=ON
```

When this is done, compile and run the program as usual:

```
$ ck compile
$ ck run --cmd_key=help
```
You can rubn on all networks

* ```ck run --cmd_key=all```


You can run on different networks:

* ``ck run --env.CK_ACL_MODEL=mobilenet (this is the default)``

You can run on different convolution methods:

* ``ck run --env.CK_ACL_CONV_METHOD=gemm (this is the default)``

