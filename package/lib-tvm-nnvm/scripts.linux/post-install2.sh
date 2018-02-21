#! /bin/bash

#
# Installation script for Caffe.
#
# See CK LICENSE for licensing details.
# See CK COPYRIGHT for copyright details.
#
# Developer(s):
# - Grigori Fursin, 2017
#

# PACKAGE_DIR
# INSTALL_DIR

echo "Post install script"

mkdir -p ${INSTALL_DIR}/install/nnvm;
mkdir -p ${INSTALL_DIR}/install/tvm;
mkdir -p ${INSTALL_DIR}/install/tvm/topi;


cp -a ${INSTALL_DIR}/src/python ${INSTALL_DIR}/install/nnvm;
cp -a ${INSTALL_DIR}/src/python ${INSTALL_DIR}/install/tvm;
cp -a ${INSTALL_DIR}/src/python ${INSTALL_DIR}/install/tvm/topi;


return 0
