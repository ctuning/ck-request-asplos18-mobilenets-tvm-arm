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
echo "Preparing vars for NNVM/TVM ..."


# Check extra stuff
EXTRA_FLAGS=""

cd ${INSTALL_DIR}/src
pwd 

if [ ${USE_OPENBLAS} == "1" ];then
   echo "Use OpenBLAS engine"
   BLAS_ENGINE="openblas";
else
  echo "Specify BLAS engine"
fi;

make -j 2\
      USE_OPENCV=${USE_OPENCV} \
      USE_BLAS=${BLAS_ENGINE} \


if [ "${?}" != "0" ] ; then
  echo "Error: cmake failed!"
  exit 1
fi

exit 1

cd ${INSTALL_DIR}/src
make; 
if [ "${?}" != "0" ] ; then
  echo "Error: cmake failed!"
  exit 1
fi
export PACKAGE_BUILD_TYPE=skip

return 0
