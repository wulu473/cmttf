#!/bin/bash

travis_retry hg clone https://bitbucket.org/eigen/eigen#3.2 /tmp/eigen
mkdir /tmp/eigen-build
cd /tmp/eigen-build
cmake . /tmp/eigen
make
sudo make install
cd -

