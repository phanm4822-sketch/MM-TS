#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>

// Single-output-spatial-position specialization of the installed PyTorch
// slow dilated Conv3D. Preserve its per-patch GEMM, layout, BF16 and beta=1;
// replace 1024 scalar bias fills per patch with one batched tensor copy.
torch::Tensor exact_patch(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda());
  TORCH_CHECK(x.device()==weight.device() && x.device()==bias.device());
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && weight.scalar_type()==at::kBFloat16 && bias.scalar_type()==at::kBFloat16);
  TORCH_CHECK(!x.requires_grad() && !weight.requires_grad() && !bias.requires_grad(),"frozen vision only");
  TORCH_CHECK(x.dim()==2 && weight.dim()==5 && bias.dim()==1);
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous());
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t b=x.size(0), n=weight.size(0), k=weight.numel()/n;
  TORCH_CHECK(x.size(1)==k && bias.size(0)==n);
  auto out=torch::empty({b,n},x.options());
  out.copy_(bias.unsqueeze(0).expand({b,n}));
  auto input=x.const_data_ptr<at::BFloat16>();
  auto w=weight.const_data_ptr<at::BFloat16>();
  auto output=out.mutable_data_ptr<at::BFloat16>();
  // Mirror PyTorch 2.9.1 CUDABlas.cpp's BF16 helper. Its internal C++
  // wrapper is hidden in the wheel, so call the same cuBLAS API directly.
  TORCH_CHECK(at::globalContext().blasPreferredBackend()!=at::BlasBackend::Cublaslt,"requires original cuBLAS backend");
  auto handle=at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t flags=CUBLAS_DEFAULT_MATH;
  if (!at::globalContext().allowBF16ReductionCuBLAS()) flags=static_cast<cublasMath_t>(flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  const float alpha=1.0f,beta=1.0f;
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle,flags));
  for (int64_t i=0;i<b;++i) {
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,n,k,&alpha,input+i*k,CUDA_R_16BF,1,w,CUDA_R_16BF,k,&beta,output+i*n,CUDA_R_16BF,1,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("exact_patch",&exact_patch);}
