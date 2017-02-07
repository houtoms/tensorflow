#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "adjust_hue_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace internal {

__global__ void adjust_hue_nhwc(const int number_elements,
                                const float * const input, float * const output, const float * const hue_delta) {

        const float delta = hue_delta[0];

	// multiply by 3 since we're dealing with contiguous RGB bytes for each pixel
	const int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 3;

	// bounds check
	if (idx > number_elements) {
		return;
	}

	// RGB to HSV
	const float r = input[idx];
	const float g = input[idx + 1];
	const float b = input[idx + 2];

	const float M = fmaxf(r, fmaxf(g, b));

	const float m = fminf(r, fminf(g, b));
	const float chroma = M - m;

	// v is the same as M
	float h = 0.0, s = 0.0;

	// hue
	if (chroma > 0.0f) {
		if (M == r) {

			const float num = (g - b) / chroma;
			const float sgn = num < 0.0f;
			const float sign = powf(-1.0f, sgn);
			h = (sgn * 6.0f + sign * fmodf(sign * num, 6.0f)) / 6.0f;

		} else if (M == g) {

			h = ((b - r) / chroma + 2.0) / 6.0f;

		} else {

			h = ((r - g) / chroma + 4.0) / 6.0f;
		}

	} else {

		h = 0.0f;
	}

	// saturation
	if (M > 0.0) {
		s = chroma / M;

	} else {

		s = 0.0f;
	}


	// hue adjustment
	h = fmodf(h + delta, 1.0f);
        if (h < 0.0f) {
          h = fmodf(1.0f + h, 1.0f);
        }

	// HSV to RGB
	const float new_h = h * 6.0f;
	const float new_chroma = M * s;
	const float x = chroma * (1.0 - fabsf(fmodf(new_h, 2.0f) - 1.0f));
	const float new_m = M - chroma;

	const bool between_0_and_1 = new_h >= 0.0 && new_h < 1;
	const bool between_1_and_2 = new_h >= 1.0 && new_h < 2;
	const bool between_2_and_3 = new_h >= 2 && new_h < 3;
	const bool between_3_and_4 = new_h >= 3 && new_h < 4;
	const bool between_4_and_5 = new_h >= 4 && new_h < 5;
	const bool between_5_and_6 = new_h >= 5 && new_h < 6;

	output[idx] = new_chroma * (between_0_and_1 || between_5_and_6) +
	                    x * (between_1_and_2 || between_4_and_5) + new_m;

	output[idx + 1] = new_chroma * (between_1_and_2 || between_2_and_3) +
	                        x * (between_0_and_1 || between_3_and_4) + new_m;

	output[idx + 2] =  new_chroma * (between_3_and_4 || between_4_and_5) +
	                         x * (between_2_and_3 || between_5_and_6) + new_m;

}
}

template <>
class AdjustHueOp<GPUDevice> : public AdjustHueOpBase {
public:
	explicit AdjustHueOp(OpKernelConstruction* const context)
		: AdjustHueOpBase(context) {}

	void DoCompute(OpKernelContext* const context,
	               const ComputeOptions& options) override {

		const Tensor* input = options.input;
		const Tensor* delta = options.delta;
		Tensor* const output = options.output;
		const int64 number_elements = input->NumElements();
		const GPUDevice &d = context->eigen_gpu_device();
		const auto stream = d.stream();

		OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

		if (number_elements > 0) {


			const CudaLaunchConfig config = GetCudaLaunchConfig(number_elements, d);
			const float * const input_data = input->flat<float>().data();
			const float * const delta_h = delta->flat<float>().data();
			float * const output_data = output->flat<float>().data();
                        const int threads_per_block = config.thread_per_block;
                        const int block_count = (number_elements + threads_per_block - 1) / threads_per_block;
			internal::adjust_hue_nhwc<<<block_count, threads_per_block, 0, stream>>>(
			    number_elements, input_data, output_data, delta_h
			);
		}
	}
};


REGISTER_KERNEL_BUILDER(Name("AdjustHue").Device(DEVICE_GPU),
                        AdjustHueOp<GPUDevice>);

}

#endif // GOOGLE_CUDA
