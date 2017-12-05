#!/bin/bash
WD=$(pwd)
cd /tmp
wget -O boost_1_64_0.tar.gz http://sourceforge.net/projects/boost/files/boost/1.64.0/boost_1_64_0.tar.gz/download
tar xzvf boost_1_64_0.tar.gz
cd boost_1_64_0
./bootstrap.sh --with-libraries=program_options,test
sudo ./boostrap install variant=release
cd $WD

