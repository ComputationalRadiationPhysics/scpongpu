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

#include <time.h>
#include <cstdlib>
#include <limits.h>
#include <cmath>
#include <iostream>
#include "../parameters.hpp"

#pragma once


/**
 * generic class to return random number of a uniform distribution between
 * start (default=0.0) and 
 * end (default=1.0)
 */ 
template<typename T, typename Seed>
class Uniformly
{
public:
  /**
   * constructor
   * set start and end value of uniform distribution
   * create seed if not done before
   */
  Uniformly(T start=0.0, T end=1.0) : start(start), end(end) 
    {
      static Seed set_seed;
    }

  /**
   * return random number
   */
  T get()
    {
      return start + ((T)rand())/((T)INT_MAX) * (end - start);
    }
  
private:
  const T start;
  const T end;
};


