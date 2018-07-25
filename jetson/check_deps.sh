#!/bin/bash

set -o pipefail
set -e

TF_PYVER=${TF_PYVER:-"2.7"}

#Install base system packages (Likely can be assumed)
sudo apt-get install -y build-essential openjdk-8-jdk zip python-pip python3-pip
pip install virtualenv && mv /usr/local/bin/virtualenv /usr/local/bin/virtualenv2
pip3 install virtualenv && mv /usr/local/bin/virtualenv /usr/local/bin/virtualenv3


#Install virtualenv, then the relevant pip packages
if [ $TF_PYVER == "2.7" ];  then
  python2 -m virtualenv2 tf_env
else
  #For python 3; Assuming python version is one of the two options
  python3 -m virtualenv3 tf_env
fi

#Activate the virtual environment; from here on, python refers to the desired version
source tf_env/bin/activate

#Check to see if bazel is installed TODO: Update
if command -v bazel 2>/dev/null; then
  echo "Bazel is installed."
else 
  echo "Bazel is not installed.  Installing now."
  mkdir ~/Bazel
  curDir=`pwd`
  cd ~/Bazel
  wget https://github.com/bazelbuild/bazel/releases/download/0.13.0/bazel-0.13.0-dist.zip
  unzip bazel-0.13.0-dist.zip
  bash compile.sh
  sudo mv output/bazel /usr/local/bin/bazel
  cd $curDir
fi

#Install required pip packages
pip install -y numpy enum34 mock


