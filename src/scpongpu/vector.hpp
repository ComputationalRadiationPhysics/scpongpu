#ifndef CUDA_VECTOR
#define CUDA_VECTOR


template<typename T>
struct cuda_vec
{
  typedef T NativeType;

  __host__ __device__ cuda_vec(T x, T y, T z)
    : x(x), y(y), z(z) {}

  __host__ __device__ cuda_vec()
    : x(0.0f), y(0.0f), z(0.0f) {}

  __host__ __device__ cuda_vec<T> operator+(const cuda_vec<T>& other) const
  {
    return cuda_vec<T>(x+other.x, y+other.y, z+other.z );
  }

  __host__ __device__ cuda_vec<T> operator-(const cuda_vec<T>& other) const
  {
    return cuda_vec<T>(x-other.x, y-other.y, z-other.z );
  }



  __host__ __device__ T operator*(const cuda_vec<T>& other) const
  {
    return x*other.x + y*other.y + z* other.z;
  }

  __host__ __device__ cuda_vec<T> operator*(const T scalar) const
  {
    return cuda_vec(scalar*x, scalar*y, scalar*z);
  }



  __host__ __device__ void operator+=(const cuda_vec<T>& other)
  {
    x += other.x;
    y += other.y;
    z += other.z;
  }

  __host__ __device__ void operator*=(const T scalar)
  {
    x *= scalar;
    y *= scalar;
    z *= scalar;
  }

  // data:
  T x;
  T y;
  T z;
  //T dummy;
};



#endif
