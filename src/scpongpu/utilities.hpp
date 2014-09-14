#pragma once

/* hide utilities behind namespace */
/* TODO better name for this namespace? */
namespace util
{

  /** 
   * generic square functions: 
   * for identical input and output domain (data type) 
   * f.e. scalar^2 = scalar
   */
  template<typename A>
  __forceinline__ __device__ __host__ A square(A a)
  {
    return a*a;
  }

  /** 
   * generic square functions: 
   * for different input and output domain (data type) 
   * f.e. vector^2 = scalar
   */
  template<typename A, typename R>
  __forceinline__ __device__ __host__ R square(A a)
  {
    return a*a;
  }

  /**
   * generic cube functions:
   * for identical input and output domain (data type) 
   */
  template<typename A>
  __forceinline__ __device__ __host__ A cube(A a)
  {
    return a*a*a;
  }

  /**
   * generic cube functions:
   * for different input and output domain (data type) 
   */
  template<typename A, typename R>
  __forceinline__ __device__ __host__ R cube(A a)
  {
    return a*a*a;
  }

}


namespace meta
{
  /* TODO need to add loop unrolling meta::For */


}


