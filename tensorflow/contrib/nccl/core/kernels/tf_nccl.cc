#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include <iostream>
#include <vector>
#include "tensorflow/contrib/nccl/third_party/nccl/nccl.h"
#include "tensorflow/contrib/nccl/third_party/nccl/nccl_manager.h"

using namespace tensorflow;

REGISTER_OP("AllReduce")
    .Attr("T: {int32, float, double}")
    .Input("to_reduce: T")
    .Input("my_rank: int32")
    .Input("all_ranks: int32")
    .Output("reduced: T");

class NCCLOp : public OpKernel {
    public:
        explicit NCCLOp(OpKernelConstruction * context) : OpKernel(context) {}

        ncclComm_t getComm(const Tensor & myRankTensor, const Tensor & allRanksTensor)
        {
            NCCLManager * nccl_mgr = NCCLManager::getManager();
            auto myRankFlat = myRankTensor.flat<int32>();
            int32 myRank = myRankFlat.data()[0];

            auto allRanks = allRanksTensor.flat<int32>();

            int nDev = allRanks.size();
            std::string devicesString = "[";
            for(int i = 0; i < nDev; ++i)
                devicesString += std::to_string(allRanks.data()[i]) + ", ";
            devicesString += "]";

            std::string myRankString = std::to_string(myRank);
            std::string commString = myRankString + ", " + devicesString;

            ncclComm_t comm = nccl_mgr->giveCommunicator(commString);
            if(comm == NULL)
            {
                ncclUniqueId id = nccl_mgr->giveNCCLId(devicesString);
                ncclResult_t status = ncclCommInitRank(&comm, nDev, id, myRank);
                nccl_mgr->storeCommunicator(commString, comm);
            }

            return comm;
        }
};
template<typename T> struct nccl_type {};
#define DEFINE_NCCL_TYPE(T, t) \
    template<> struct nccl_type<T> { static const ncclDataType_t value = t; }
DEFINE_NCCL_TYPE(signed char, ncclChar);
DEFINE_NCCL_TYPE(int32,       ncclInt);
DEFINE_NCCL_TYPE(float,       ncclFloat);
DEFINE_NCCL_TYPE(double,      ncclDouble);
DEFINE_NCCL_TYPE(int64,       ncclInt64);
DEFINE_NCCL_TYPE(uint64,      ncclUint64);

template <typename T>
class AllReduceOp : public NCCLOp {
    public:
        explicit AllReduceOp(OpKernelConstruction * context) : NCCLOp(context) {}

        void Compute(OpKernelContext * context) override {

            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<T>();
            const T * input_data = input.data();
            const int N = input.size();

            Tensor * output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
            auto output = output_tensor->template flat<T>();

            const Tensor & myRankTensor = context->input(1);
            auto myRankFlat = myRankTensor.flat<int32>();
            int32 myRank = myRankFlat.data()[0];

            const Tensor & allRanksTensor = context->input(2);

            ncclComm_t comm = getComm(myRankTensor, allRanksTensor);

            auto * dc = context->op_device_context();
            auto * s = dc->stream();
            perftools::gputools::cuda::CUDAStream * cudastream = perftools::gputools::cuda::AsCUDAStream(s);
            CUstream stream = cudastream->cuda_stream();

            ncclAllReduce(input_data, output.data(), N, nccl_type<T>::value, ncclSum, comm, stream);
        }
};

#define REGISTER_ALLREDUCE(type) \
    REGISTER_KERNEL_BUILDER(Name("AllReduce").Device(DEVICE_GPU).TypeConstraint<type>("T").HostMemory("my_rank").HostMemory("all_ranks"), AllReduceOp<type>);

REGISTER_ALLREDUCE(int32);
REGISTER_ALLREDUCE(float);
REGISTER_ALLREDUCE(double);

#undef REGISTER_ALLREDUCE

REGISTER_OP("Bcast")
    .Attr("T: {int32, float, double}")
    .Input("to_bcast: T")
    .Input("from_rank: int32")
    .Input("my_rank: int32")
    .Input("all_ranks: int32")
    .Output("bcasted: T");

template <typename T>
class BcastOp : public NCCLOp {
    public:
        explicit BcastOp(OpKernelConstruction * context) : NCCLOp(context) {}

        void Compute(OpKernelContext * context) override {
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<T>();
            T * input_data = (T *) input.data();
            const int N = input.size();

            const Tensor & fromRankTensor = context->input(1);
            auto fromRankFlat = fromRankTensor.flat<int32>();
            int32 fromRank = fromRankFlat.data()[0];

            const Tensor & myRankTensor = context->input(2);
            auto myRankFlat = myRankTensor.flat<int32>();
            int32 myRank = myRankFlat.data()[0];

            Tensor * output_tensor = NULL;
            T * output_data = NULL;
            if(myRank != fromRank)
            {
                OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
                auto output = output_tensor->template flat<T>();
                output_data = output.data();
            }


            const Tensor & allRanksTensor = context->input(3);

            ncclComm_t comm = getComm(myRankTensor, allRanksTensor);

            auto * dc = context->op_device_context();
            auto * s = dc->stream();
            perftools::gputools::cuda::CUDAStream * cudastream = perftools::gputools::cuda::AsCUDAStream(s);
            CUstream stream = cudastream->cuda_stream();

            if(myRank == fromRank)
            {
                ncclBcast(input_data, N, nccl_type<T>::value, fromRank, comm, stream);
                context->set_output(0, context->input(0));
            }
            else
            {
                ncclBcast(output_data, N, nccl_type<T>::value, fromRank, comm, stream);
            }
        }
};

#define REGISTER_BCAST(type) \
    REGISTER_KERNEL_BUILDER(Name("Bcast").Device(DEVICE_GPU).TypeConstraint<type>("T").HostMemory("from_rank").HostMemory("my_rank").HostMemory("all_ranks"), BcastOp<type>);

REGISTER_BCAST(int32);
REGISTER_BCAST(float);
REGISTER_BCAST(double);

#undef REGISTER_BCAST
