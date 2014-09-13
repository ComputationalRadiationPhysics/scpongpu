#include <time.h>
#include <cstdlib>
#include <limits.h>
#include <cmath>
#include <iostream>
#include "parameters.hpp"

#pragma once

/**
 * class to set the seed once for all random calls
 * currently use fixed seed to reproduce physics
 */
class Monte_Carlo_Switch
{
public:
  Monte_Carlo_Switch()  
    {
      /* currently fixed seed to reproduce physics */
      srand( 12 /*time(NULL)*/ );  /* set seed once */
      std::cout << " seed was set " << std::endl;
      /* TODO make seed time, machine etc. dependent
         TODO select seed type in via parameter file
         issue #6
       */         
    } 
};


/**
 * generic class to return random number of a uniform distribution between
 * start (default=0.0) and 
 * end (default=1.0)
 */ 
template<typename T>
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
      static Monte_Carlo_Switch set_seed;
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


/**
 * generic class to return random number according to a Gaussian 
 * distribution around mu (default=0.0) with sigma (default=1.0)
 */
template <typename T>
class Gauss_1D
{
public:
  /**
   * constructor
   * set center (mu) and spread (sigma) of Gaussian distribution
   */
  Gauss_1D(T mu=0.0, T sigma=1.0) 
    : mu(mu), sigma(sigma)
    {
      /* We use a brute force random number generation as described by Blobel.
       * For that, we need a range of possible x-values.
       * Since the Gaussian distribution rages from - to + infinity,
       * we have to limit the range. 
       * TODO? use faster algorithm for Gaussian distribution? */
      min = mu - 5.0 * sigma; 
      max = mu + 5.0 * sigma; /* contains 99.99994% of all values */
      /* however this should fit to the problem
       * for 10^8 particles wider range */
      
      /* any number larger than the max of the Gaussian distribution: */
      const T supremum = function(mu) * 1.01;

      /* uniform random number generated from 0-supremum: */
      uni = new Uniformly<T>(0.0, supremum);

      /* uniform random number generator for x-values under Gaussian curve: */
      in_range = new Uniformly<T>(min, max);
    }

  /**
   * advanced constructor
   * set center (mu) and spread (sigma) of Gaussian distribution
   * additionally: set on x-range for Gaussian distribution (min, max)
   *
   * only use this constructor, if you know what you are doing
   * too large values will coast a lot of computing power
   * out of range values will not reproduce a Gaussian distribution
   * asymmetry causes no problem
   */
  Gauss_1D(T mu, T sigma, T min, T max) 
    : mu(mu), sigma(sigma), min(min), max(max)
    {
      /* for details see simpler constructor */
      const T supremum = function(mu) * 1.01;
      uni = new Uniformly<T>(0.0, supremum);
      in_range = new Uniformly<T>(min, max);
    }


  /**
   * destructor
   */
  ~Gauss_1D()
    {
      /* free uniform number generators */
      delete uni;
      delete in_range;
    }

  /**
   * return random number based on Gaussian distribution
   *
   * it is based on a brute force algorithm - there is definitely a better and 
   * faster way to do this, f.e. see Bloebel
   */
  T get()
    {
      T u_1; /* x-value to evaluate p1=Gauss(u_1) at */
      T v;   /* the probability p1 of the Gaussian distribution at u_1 */
      T u_2; /* a random number between 0.0 and supremum */
      do
        {
          u_1 = in_range->get(); /* rand [-5sigma, +5sigma] */
          v = function(u_1); /* p1 = Gauss(u_1) */
          u_2 = uni->get(); /* p2 = rand [0.0, 1.01*maxGauss] */
        }
      while (v < u_2);
      /* as long as p_1 < p2, try again */

      /* if above relation is fulfilled, u_1 is your random number
       * statistically it follows the distribution described in function()
       * here: Gaussian distribution */
      return u_1;
    }

private:
  const T mu; /* center of Gaussian distribution */
  const T sigma; /* standard deviation */
  T min; /* min value for x */
  T max; /* max value for x */
  Uniformly<T>* in_range; /* uniform distribution from [min, max] */
  Uniformly<T>* uni; /* uniform distribution [0.0, supremum] */

  /**
   * function describing the probability distribution
   * here: Gaussian distribution
   * (a normalization is not needed)
   */
  T function(T x)
    {
      return 1.0/(sqrt(2*M_PI) * sigma) * exp(-0.5*(x-mu)*(x-mu)/(sigma*sigma));
    }
};


/**
 * generic class to generate Maxwell-Boltzmann distributed velocities
 * for a single dimension
 */
template<typename T>
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
      distribution = new Gauss_1D<T>(0.0, sigma);
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
  Gauss_1D<T>* distribution; /* Gaussian random number generator */
  
};



