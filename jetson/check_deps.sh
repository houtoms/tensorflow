#!/bin/bash

#TODO Ask user if the packages should be installed
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

if sudo dpkg -V python-pip 2>/dev/null; then
  echo "Python-pip package is installed."
else
  echo "Python-pip package is not installed.  Installing now."
  echo "y" | sudo apt-get install python-pip
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

#Check to see if numpy is installed
if pip list | grep numpy 2>/dev/null; then
  echo "numpy is installed"
else
  echo "numpy is not installed.  Installing now."
  echo "y" | sudo pip install numpy
fi

#Check 


