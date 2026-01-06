#include <mudnn.h>
#include "DeviceThreadHandles.h"

using muTensor = ::musa::dnn::Tensor;
using muHandle = ::musa::dnn::Handle;
using mudnnHandle_t = ::musa::dnn::Handle*;

void CreateMuDNNHandle(mudnnHandle_t* handle) {
  int device;
  musaGetDevice(&device);
  *handle = new muHandle(device);
}

void DestroyMuDNNHandle(mudnnHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the musa context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. Not destroy the handle as a workaround.
}

using MudnnPoolType = at::musa::DeviceThreadHandlePool<
    mudnnHandle_t,
    CreateMuDNNHandle,
    DestroyMuDNNHandle>;


::musa::dnn::Handle& GetMudnnHandle(musaStream_t stream) {
  int device;
  musaGetDevice(&device);

  static auto pool = std::make_shared<MudnnPoolType>();
  thread_local std::unique_ptr<MudnnPoolType::PoolWindow> myPoolWindow(
      pool->NewPoolWindow());

  mudnnHandle_t handle = myPoolWindow->reserve(device);
  handle->SetStream(stream);
  return *handle;
}

void SetMUTensorDType(infiniDtype_t dtype, muTensor& m_t) {
  switch (dtype) {
    case INFINI_DTYPE_F16:
      m_t.SetType(muTensor::Type::HALF);
      break;
    case INFINI_DTYPE_F32:
      m_t.SetType(muTensor::Type::FLOAT);
      break;
    default:
      std::cout << "SetMUTensorDType Unsupported tensor dtype: " << dtype << std::endl;
      throw;
  }
}

void SetMUTensorAddr(const void* addr, muTensor& m_t) {
  m_t.SetAddr(addr);
}

void ConfigFormat(const void* t, muTensor& mt, int64_t dim, const int64_t* sizes, const int64_t* strides, bool permute_if_not_contiguous = true) {
  muTensor::Format mudnn_format = muTensor::Format::NCHW;

  mt.SetFormat(mudnn_format);
  mt.SetNdInfo(dim, sizes, strides);
}

muTensor CreateMUTensor(const void* t, infiniDtype_t datatype, int64_t dim, const int64_t* sizes, const int64_t* strides, bool permute_if_not_contiguous = true) {
  muTensor rst;
  SetMUTensorDType(datatype, rst);
  SetMUTensorAddr(t, rst);
  ConfigFormat(t, rst, dim, sizes, strides, permute_if_not_contiguous);
  return rst;
}

void SwishGluOut(const void* input, const void* output, infiniDtype_t datatype, musaStream_t stream, int64_t dim, 
  const int64_t* input_sizes, const int64_t* output_sizes, 
  const int64_t* input_strides, const int64_t* output_strides) {

  auto mt_input = CreateMUTensor(input, datatype, dim, input_sizes, input_strides);
  auto mt_output = CreateMUTensor(output, datatype, dim, output_sizes, output_strides);

  muHandle& h = GetMudnnHandle(stream);

  ::musa::dnn::SwiGlu op;

  auto status = op.Run(h, mt_output, mt_input);
  if (status != ::musa::dnn::Status::SUCCESS) {
    std::cout << "MUDNN failed in: Run" << std::endl;
  }

}

void SwishGlu(const void* input, const void* output, infiniDtype_t datatype, musaStream_t stream, int64_t dim, 
  const int64_t* input_sizes, const int64_t* output_sizes, 
  const int64_t* input_strides, const int64_t* output_strides) {
  SwishGluOut(input, output, datatype, stream, dim, input_sizes, output_sizes, input_strides, output_strides);
}