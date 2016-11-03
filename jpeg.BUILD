# Description:
#   The Independent JPEG Group's JPEG runtime library.
licenses(["notice"])  # custom notice-style license, see LICENSE

HEADERS = [
    "cderror.h",
    "cdjpeg.h",
    "jconfig.h",
    "jdct.h",
    "jerror.h",
    "jinclude.h",
    "jmemsys.h",
    "jmorecfg.h",
    "jpegint.h",
    "jpeglib.h",
    "jversion.h",
    "transupp.h",
    "jsimd.h",
#    "md5.h"
]

prefix_dir = "libjpeg-turbo-1.5.1"

genrule(
    name = "assemble",
    outs = ["libjpeg.a"] + HEADERS,
    cmd = "pushd external/jpeg_archive/libjpeg-turbo-1.5.1  && workdir=$$(mktemp -d -t tmp.XXXXXXXXXX) && cp -a * $$workdir && pushd $$workdir && autoreconf -fiv && CFLAGS=\"-fPIC -O3\" ./configure --with-jpeg8 && make clean && make && popd && popd && cp $$workdir/.libs/lib*jpeg* $(@D) && cp $$workdir/*.h $(@D) && rm -rf $$workdir",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "jpeg",
    srcs = ["libjpeg.a"], 
    hdrs = HEADERS, # glob(["**/*.h"]), 
    includes = [prefix_dir],
    visibility = ["//visibility:public"],
    linkstatic = 1
)

