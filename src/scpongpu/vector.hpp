#pragma once


/**
 * Template 3D vector with T as data type
 * used as vector class on CUDA devices
 */
template<typename T>
struct cuda_vec
{
  typedef T NativeType;

  /**
   * CPU/Host constructor if values are given
   */
  __host__ __device__ cuda_vec(T x, T y, T z)
    : x(x), y(y), z(z) {}


  /**
   * CPU/Host constructive if values are not given
   */
  __host__ __device__ cuda_vec()
    : x(0.0f), y(0.0f), z(0.0f) {}


  /**
   * overload + operator for component-wise addition
   */
  __host__ __device__ cuda_vec<T> operator+(const cuda_vec<T>& other) const
  {
    return cuda_vec<T>(x+other.x, y+other.y, z+other.z );
  }


  /**
   * overload - operator for component-wise subtraction
   */
  __host__ __device__ cuda_vec<T> operator-(const cuda_vec<T>& other) const
  {
    return cuda_vec<T>(x-other.x, y-other.y, z-other.z );
  }


  /**
   * overload * operator for scalar product with other vector
   */
  __host__ __device__ T operator*(const cuda_vec<T>& other) const
  {
    return x*other.x + y*other.y + z* other.z;
  }

  /**
   * overload * operator for scalar product with scalar
   */
  __host__ __device__ cuda_vec<T> operator*(const T scalar) const
  {
    return cuda_vec(scalar*x, scalar*y, scalar*z);
  }


  /**
   * overload += operator for component-wise add-assign
   */
  __host__ __device__ void operator+=(const cuda_vec<T>& other)
  {
    x += other.x;
    y += other.y;
    z += other.z;
  }

  /**
   * overload *= operator for assign the product with a scalar
   */
  __host__ __device__ void operator*=(const T scalar)
  {
    x *= scalar;
    y *= scalar;
    z *= scalar;
  }

  /* TODO: could this be private? */
  /**
   * data for 3D vector 
   */
  T x;
  T y;
  T z;

};




