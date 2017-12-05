#!/bin/bash
WD=$(pwd)
cd /tmp
wget -O boost_1_64_0.tar.gz http://sourceforge.net/projects/boost/files/boost/1.64.0/boost_1_64_0.tar.gz/download
tar -xf boost_1_64_0.tar.gz
cd boost_1_64_0
./bootstrap.sh --with-libraries=program_options,test
./b2 -d0 variant=release
sudo ./b2 -d0 install
cd $WD

