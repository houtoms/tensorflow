#!/bin/bash

set -o pipefail
set -e

TF_PYVER=${TF_PYVER:-"2.7"}

#Check if base system packages are installed
if sudo dpkg -V build-essential 2>/dev/null; then
  echo "build-essential package is installed."
else
  echo "build-essential package is not installed.  Installing now."
  echo "y" | sudo apt-get install build-essential
fi  

if sudo dpkg -V openjdk-8-jdk 2>/dev/null; then
  echo "OpenJDK-8-jdk package is installed."
else
  echo "OpenJDK-8-jdk package is not installed.  Installing now."
  echo "y" | sudo apt-get install openjdk-8-jdk
fi

if sudo dpkg -V zip 2>/dev/null; then
  echo "zip package is installed."
else
  echo "zip package is not installed.  Installing now."
  echo "y" | sudo apt-get install zip
fi

if [ $TF_PYVER == "2.7" ];  then
  echo "2.7"
  if sudo dpkg -V python-pip 2>/dev/null; then
    echo "Python-pip package is installed."
  else
    echo "Python-pip package is not installed.  Installing now."
    echo "y" | sudo apt-get install python-pip
  fi
else
  if [ $TF_PYVER != "3.5" ]; then
    echo "Python version must be either 2.7 or 3.5.  Exiting now."
    exit 1
  else  
    echo "3.5"
    #For python 3
    if sudo dpkg -V python3-pip 2>/dev/null; then
      echo "Python3-pip package is installed."
    else
      echo "Python3-pip package is not installed.  Installing now."
      echo "y" | sudo apt-get install python3-pip
    fi
  fi
fi

#Check to see if bazel is installed
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


if [ $TF_PYVER == "2.7" ]; then
  #Check to see if numpy is installed
  if pip list | grep numpy 2>/dev/null; then
    echo "numpy is installed"
  else
    echo "numpy is not installed.  Installing now."
    echo "y" | sudo pip install numpy
  fi

  #Check for enum34
  if pip list | grep enum34 2>/dev/null; then
    echo "enum34 is installed"
  else
    echo "enum34 is not installed.  Installing now."
    echo "y" | sudo pip install enum34
  fi

  #Check for mock
  if pip list | grep mock 2>/dev/null; then
    echo "mock is installed"
  else
    echo "mock is not installed.  Installing now."
    echo "y" | sudo pip install mock
  fi
else
  #Python 3 versions of required packages
  if pip3 list | grep numpy 2>/dev/null; then
    echo "numpy is installed"
  else
    echo "numpy is not installed.  Installing now."
    echo "y" | sudo pip3 install numpy
  fi
  
  if pip3 list | grep enum34 2>/dev/null; then
    echo "enum34 is installed"
  else
    echo "enum34 is not installed.  Installing now."
    echo "y" | sudo pip3 install enum34
  fi

  if pip3 list | grep mock 2>/dev/null; then
    echo "mock is installed"
  else
    echo "mock is not installed.  Installing now."
    echo "y" | sudo pip3 install mock
  fi
fi
