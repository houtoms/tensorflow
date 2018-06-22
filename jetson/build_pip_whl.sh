#!/bin/bash

#This script should:
#	1) Run the configure script
#	2) Compile tensorflow with bazel
#	3) Build the pip whl
#	4) Install the tensorflow package with pip

#TODO automatically input the options for the configuration script:
#Location of python: (default)
Result="/usr/bin/python\n"
#Desired python library: (default)
R2="/usr/local/lib/python2.7/dist-packages\n"
Result=$Result$R2
#Jemalloc: (No) 
R2="n\n"
Result=$Resulti$R2
#Google cloud platform support: (no)
Result=$Result$R2
#Hadoop: (no)
Result=$Result$R2
#Amazon S3: (no)
Result=$Result$R2
#Apache Kafka: (no)
Result=$Result$R2
#XLA JIT: (no)
Result=$Result$R2
#GDR: (no)
Result=$Result$R2
#VERBS: (no)
Result=$Result$R2
#OpenCL SYCL: (no)
Result=$Result$R2
#CUDA: (yes)
R2="y\n"
Result=$Result$R2
#CUDA version: (9)
R2="9\n"
Result=$Result$R2
#cuda toolkit location: (default)
R2="\n"
Result=$Result$R2
#cuDNN version: (default)
Result=$Result$R2
#cuDNN location: (default)
Result=$Result$R2
#TensorRT support: (yes)
R2="y\n"
Result=$Result$R2
#TensorRT location: (default)
R2="\n"
Result=$Result$R2
#NCCL version:  (for now, default) 
Result=$Result$R2
#cuda compute capability: (6.2)
R2="6.2\n"
Result=$Result$R2
#clang: (no)
R2="n\n"
Result=$Result$R2
#specify gcc: (default)
R2="\n"
Result=$Result$R2
#MPI support: (no)
R2="n\n"
Result=$Result$R2
#optimization flags: (default)
R2="\n"
Result=$Result$R2
#interactively configure workspace for Android: (no)
R2="n\n"
Result=$Result$R2



#Result="\n\nn\nn\nn\nn\nn\nn\nn\nn\nn\ny\n9\n\n\n\ny\n\n\n6.2\nn\n\nn\n\nn\n"
#echo $Result
echo -e $Result | ../configure




bazel build --config=opt --config=cuda ../tensorflow/tools/pip_package/build_pip_package
#TODO check result of build command, if the build failed throw error

#bash ../tensorflow/tools/pip_package/build_pip_package.sh /tmp/tensorflow_pkg


#TODO check if tensorflow is installed, act based thereon
#sudo -H pip install /tmp/tensorflow_pkg/*






