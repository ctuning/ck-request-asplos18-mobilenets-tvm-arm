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
EXTRA_FLAGS=""

cd ${INSTALL_DIR}/src

make -j 2 \
      USE_OPENCV=${USE_OPENCV} \
      USE_BLAS=openblas \


#export PACKAGE_BUILD_TYPE=skip

return 0
