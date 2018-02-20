#! /bin/bash

#
# Installation script for Caffe.
#
# See CK LICENSE for licensing details.
# See CK COPYRIGHT for copyright details.
#
# Developer(s):
# - Grigori Fursin, 2015;
# - Anton Lokhmotov, 2016.
#

# PACKAGE_DIR
# INSTALL_DIR

echo "**************************************************************"
echo "Preparing vars for MxNet ..."

# Check extra stuff
export MXNET_LIB_DIR=${INSTALL_DIR}/lib
export CK_MAKE_BEFORE="USE_OPENCV=1 USE_BLAS=openblas"
cd ${INSTALL_DIR}/src

make -j 2 \
      USE_OPENCV=${USE_OPENCV} \
      USE_BLAS=openblas \


return 0
