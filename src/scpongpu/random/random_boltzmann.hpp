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
#include "random_gauss.hpp"
#include "../parameters.hpp"

#pragma once


/**
 * generic class to generate Maxwell-Boltzmann distributed velocities
 * for a single dimension
 */
template<typename T, typename Seed>
class Maxwell_Boltzmann
{
public:
  /**
   * constructor
   * initialize Boltzmann distribution with temperate and particle mass
   */
  Maxwell_Boltzmann(T Temperatur, T mass)
    : Temperatur(Temperatur), mass(mass), k_Boltzmann(parameters::boltzmann)
    {
      /* Boltzmann distribution for each dimension is just a Gaussian distribution:
       * convert temperature and mass to standard deviation */
      const T sigma = sqrt(k_Boltzmann * Temperatur / mass);
      /* use generic Gaussian random number generator to build Boltzmann random number generator */
      distribution = new Gauss_1D<T, Seed>(0.0, sigma);
     }

  /**
   * destructor
   */
  ~Maxwell_Boltzmann()
  {
    /* free Gaussian random number generator */
    delete distribution; 
  } 

  /**
   * returns one dimensional velocities according to 
   * the Maxwell-Boltzmann distribution 
   */
  T get()
    {
      return distribution->get();
    }

private:
  const T k_Boltzmann; /* Boltzmann constant (SI units) */
  const T Temperatur; /* temperature in Kelvin */
  const T mass; /* mass in kg */
  Gauss_1D<T, Seed>* distribution; /* Gaussian random number generator */
  
};



