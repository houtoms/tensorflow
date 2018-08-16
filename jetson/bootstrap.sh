#!/bin/bash

# Create and enable swapfile
if [ ! -f /swapfile ]; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  echo "/swapfile none swap defaults 0 0" >> /etc/fstab
fi
swapon -a

# Set startup script to boost clocks to max
if [ -z "$(grep jetson_clocks.sh /etc/rc.local)" ]; then
  sed -i 's,^exit 0$,/home/nvidia/jetson_clocks.sh\nexit 0,' /etc/rc.local
fi
# Set clocks to max immediately as well
/home/nvidia/jetson_clocks.sh
systemctl disable ondemand nvpmodel

# Install base system packages
apt-get update && apt-get install -y build-essential openjdk-8-jdk zip python-pip python3-pip libfreetype6-dev libpng12-dev libjpeg8-dev
pip install virtualenv
pip3 install virtualenv

# Install bazel
BAZEL_VERSION=0.15.0
mkdir -p /tmp/bazel
pushd /tmp/bazel
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-dist.zip
unzip bazel-${BAZEL_VERSION}-dist.zip
bash compile.sh
cp output/bazel /usr/local/bin/bazel
popd


#Exit script if not CI
CI_BOOTSTRAP=${CI_BOOTSTRAP:-1}
if [ $CI_BOOTSTRAP -eq 0 ]; then
  exit 0
fi

# Install golang
apt-get install -y curl wget apt-transport-https
curl -L https://storage.googleapis.com/golang/go1.10.2.linux-arm64.tar.gz | tar -C /usr/local -xzf -

# Install protobuf compiler
PROTOBUF_VERSION=3.4.0 && \
wget https://github.com/google/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz && \
tar -xzf v${PROTOBUF_VERSION}.tar.gz && \
cd protobuf-${PROTOBUF_VERSION} && \
./autogen.sh && \
./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 > /dev/null && \
rm -rf /protobuf-${PROTOBUF_VERSION}


# Install gitlab-ci-multi-runner
export GOPATH=$HOME/Go
export PATH=$PATH:$GOPATH/bin:/usr/local/go/bin
mkdir -p $GOPATH
go get gitlab.com/gitlab-org/gitlab-runner
pushd $GOPATH/src/gitlab.com/gitlab-org/gitlab-runner/
git checkout v11.1.0
make -j$(nproc) deps
make -j$(nproc) install
cp $GOPATH/bin/gitlab-runner /usr/local/bin
popd

# Register the gitlab runner
pushd $HOME
git clone https://gitlab-dl.nvidia.com/devops/gitlab-runner.git
cd gitlab-runner
./run-jetson.sh 0 TX2
popd

# Clean up startup -- no GUI, no docker
systemctl set-default multi-user.target
systemctl stop lightdm
chmod -x /usr/sbin/lightdm
systemctl stop docker
systemctl disable docker
ifdown docker0

