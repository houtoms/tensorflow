#!/bin/bash

# Set up NFS
apt-get update && apt-get install -y --no-install-recommends nfs-common
if [ -z "$(grep 10.31.241.20 /etc/fstab)" ]; then
  echo "10.31.241.20:/netapp2_labA_dgx /mnt/shared nfs rw,noatime,rsize=65536,wsize=65536,nolock,hard,intr,proto=tcp,timeo=600,retrans=3,sec=sys,vers=3,_netdev 0 0" >> /etc/fstab
fi
mkdir -p /mnt/shared
mount -a -t nfs
ln -s /mnt/shared/dldata /data || true


# Create and enable swapfile
if [ ! -f /swapfile ]; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  echo "/swapfile none swap defaults 0 0" >> /etc/fstab
fi
swapon -a


# Set startup script to boost clocks to max
touch /etc/rc.local
if [ -z "$(grep jetson_clocks.sh /etc/rc.local)" ]; then
  if test "$(wc -l /etc/rc.local | cut -d ' ' -f 1) -gt 0" && grep "^exit " /etc/rc.local &>/dev/null; then
    sed -i 's,^\(exit .*\)$,/home/nvidia/jetson_clocks.sh\n\1,' /etc/rc.local
  else
    echo "/home/nvidia/jetson_clocks.sh" >> /etc/rc.local
  fi
fi
if ! grep "^#!" /etc/rc.local &>/dev/null; then
  LOCAL=$(cat /etc/rc.local)
  echo '#!/bin/bash' > /etc/rc.local
  echo "${LOCAL}" >> /etc/rc.local
fi
chmod +x /etc/rc.local
# Set clocks to max immediately as well
systemctl stop ondemand nvpmodel
/home/nvidia/jetson_clocks.sh
systemctl disable ondemand nvpmodel

# Install base system packages
apt-get update && apt-get install libpng12-dev || echo "libpng12-dev not found" && 
	apt-get install -y build-essential openjdk-8-jdk zip python-pip python3-pip libfreetype6-dev libjpeg8-dev libhdf5-dev
pip install virtualenv
pip3 install virtualenv

# Install bazel
BAZEL_VERSION=0.20.0
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

# Determine machine type
if [ -e "/sys/devices/soc0/family" ]; then
  if [ -e "/sys/devices/soc0/machine" ]; then
    machine="`cat /sys/devices/soc0/machine`"
  fi
elif [ -e "/proc/device-tree/compatible" ]; then
  if [ -e "/proc/device-tree/model" ]; then
    machine="$(tr -d '\0' < /proc/device-tree/model)"
  fi
fi
case $machine in
  quill)         MODELNAME="TX2";;
  jetson-xavier) MODELNAME="XAVIER";;
  *)             echo "Unknown machine type $machine"; exit 1;;
esac

# Register the gitlab runner
pushd $HOME
git clone https://gitlab-master.nvidia.com/dl/devops/gitlab-runner.git
cd gitlab-runner
./run-jetson.sh 0 ${MODELNAME}
popd

# Clean up startup -- no GUI, no docker
systemctl set-default multi-user.target
systemctl isolate multi-user.target
systemctl stop lightdm
chmod -x /usr/sbin/lightdm
systemctl stop docker
systemctl disable docker
ifdown docker0

