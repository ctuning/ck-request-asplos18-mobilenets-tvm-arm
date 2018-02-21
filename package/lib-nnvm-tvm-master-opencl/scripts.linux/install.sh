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
echo "Preparing vars for TVM ..."


# Check extra stuff
EXTRA_FLAGS=""

cd ${INSTALL_DIR}/src/tvm
pwd 
make -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
      USE_OPENCL=${USE_OPENCL}\
      LLVM_CONFIG=${CK_LLVM_CONFIG} \

if [ "${?}" != "0" ] ; then
  echo "Error: make failed!"
  exit 1
fi
cd ${INSTALL_DIR}/src

echo "**************************************************************"
echo "Preparing vars for NVVM ..."

make; 
if [ "${?}" != "0" ] ; then
  echo "Error: cmake failed!"
  exit 1
fi
#export PACKAGE_BUILD_TYPE=skip

return 0
