/**
 * Copyright 2014  Richard Pausch
 *
 * This file is part of SCPonGPU.
 *
 * SCPonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SCPonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SCPonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

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


