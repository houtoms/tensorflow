#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#ifndef TENSORFLOW_CORE_KERNELS_ADJUST_HUE_OP_H_
#define TENSORFLOW_CORE_KERNELS_ADJUST_HUE_OP_H_

namespace tensorflow {

class AdjustHueOpBase : public OpKernel {
protected:
  AdjustHueOpBase(OpKernelConstruction* context) : OpKernel(context) {}

  struct ComputeOptions {
    const Tensor* input;
    const Tensor* delta;
    Tensor* output;
    int64 channel_count;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

  void Compute(OpKernelContext* context) override {

    const Tensor& input = context->input(0);
    const Tensor& delta = context->input(1);

    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta.shape()),
                errors::InvalidArgument("delta must be scalar: ",
                                        delta.shape().DebugString()));

    const auto channels = input.dim_size(input.dims() - 1);

    OP_REQUIRES(
      context, channels == 3,
      errors::InvalidArgument("input must have 3 channels but instead has ",
                              channels, " channels."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64 channel_count = input.NumElements() / channels;
      ComputeOptions options;
      options.input = &input;
      options.delta = &delta;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};


template <class Device>
class AdjustHueOp;

}

#endif // TENSORFLOW_CORE_KERNELS_ADJUST_HUE_OP_H_
