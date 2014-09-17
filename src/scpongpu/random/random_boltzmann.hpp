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



