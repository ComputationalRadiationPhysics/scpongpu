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

#pragma once

/**
 * class to set the same seed once for all random calls
 * use fixed seed to reproduce physics
 */
class Monte_Carlo_Switch_fixed
{
public:
  Monte_Carlo_Switch_fixed()  
    {
      const unsigned int seed = 12;
      srand( seed );  /* set seed once */
      std::cout << " seed was set to: " << seed 
                << " (fixed value)" << std::endl;
    } 
};


/**
 * class to set the seed randomly once for all random calls
 * no reproduction of exact results
 */
class Monte_Carlo_Switch_random
{
public:
  Monte_Carlo_Switch_random()  
    {
      const unsigned int seed = time(NULL)*::getpid();
      srand( seed );  /* set seed once */
      std::cout << " seed was set to: " << seed 
                << " (randomly set)" << std::endl;
    } 
};






