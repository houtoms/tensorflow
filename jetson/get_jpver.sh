#!/bin/bash

# Determine JetPack version for wheel naming
L4TVER=$(head /etc/nv_tegra_release  | grep REVISION: | cut -d ' ' -f 2,5 | sed 's/^\(.*\) \(.*\),.*$/\1.\2/')
case $L4TVER in
  R28.2.1) JPVER="jp33";;
  R31.0.1) JPVER="jp40";;
  R31.0.2) JPVER="jp41";;
  R31.1.0) JPVER="jp411";;
  R32.0.0) JPVER="jp42";;
  *) echo "unknown JetPack version!"; exit 1;;
esac

echo "$JPVER"
