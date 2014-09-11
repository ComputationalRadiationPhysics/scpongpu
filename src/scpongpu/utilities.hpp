

#ifndef UTILITIES_CUDA
#define UTILITIES_CUDA

namespace util
{

  // generic square functions:

  template<typename A>
  __forceinline__ __device__ __host__ A square(A a)
  {
    return a*a;
  }

  template<typename A, typename R>
  __forceinline__ __device__ __host__ R square(A a)
  {
    return a*a;
  }

  // generic cube functions:

  template<typename A>
  __forceinline__ __device__ __host__ A cube(A a)
  {
    return a*a*a;
  }

  template<typename A, typename R>
  __forceinline__ __device__ __host__ R cube(A a)
  {
    return a*a*a;
  }

}


namespace meta
{
  // need to add loop unrolling meta::For


}

#endif
