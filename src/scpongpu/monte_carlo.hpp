#include <time.h>
#include <cstdlib>
#include <limits.h>
#include <cmath>
#include <iostream>
#include "parameters.hpp"

#ifndef MONTECARLO_RPAUSCH
#define MONTECARLO_RPAUSCH

class Monte_Carlo_Switch
{
public:
  Monte_Carlo_Switch()  
    {
      srand( 12 /*time(NULL)*/ );  // set seed once 
      std::cout << " seed was set " << std::endl;
    } 
};


template<typename T>
class Uniformly
{
public:
  Uniformly(T start=0.0, T end=1.0) : start(start), end(end) 
    {
      static Monte_Carlo_Switch set_seed;
    }

  T get()
    {
      return start + ((T)rand())/((T)INT_MAX) * (end - start);
    }
  

private:
  const T start;
  const T end;
};


template <typename T>
class Gauss_1D
{
public:
  Gauss_1D(T mu=0.0, T sigma=1.0) 
    : mu(mu), sigma(sigma)
    {
      min = mu - 5.0 * sigma; // contains 99.99994% of all values
      max = mu + 5.0 * sigma; // however this should fit to the problem
                              // for 10^8 particles wider range
                              // better faster algorithm
      const T supremum = function(mu) * 1.01;
      uni = new Uniformly<T>(0.0, supremum);
      in_range = new Uniformly<T>(min, max);
    }

  Gauss_1D(T mu, T sigma, T min, T max) 
    : mu(mu), sigma(sigma), min(min), max(max)
    {
      //only use this constructor, if you know what you are doing
      //too large values will coast a lot of computing power
      //out of range values will not reproduce a gaussian distribution
      //asymetry causes no problem
      const T supremum = function(mu) * 1.01;
      uni = new Uniformly<T>(0.0, supremum);
      in_range = new Uniformly<T>(min, max);
    }


  ~Gauss_1D()
    {
      delete uni;
      delete in_range;
    }

  T get()
    {
      // this is a brute  force algoritm - there is definitly a better and
      // faster way to do this, f.e. Bloebel discribes it
      // however, right now, I don't have a copy of Bloebel on my desk
      // therefore it is the slow algorithm I remember
      
      T v;
      T u_1;
      T u_2;
      do
	{
	  u_1 = in_range->get();
	  v = function(u_1);
	  u_2 = uni->get();
	}
      while (v < u_2);

      return u_1;
    }

private:
  const T mu;
  const T sigma;
  T min;
  T max;
  Uniformly<T>* uni;
  Uniformly<T>* in_range;

  T function(T x)
    {
      return 1.0/(sqrt(2*M_PI) * sigma) * exp(-0.5*(x-mu)*(x-mu)/(sigma*sigma));
    }

};

template<typename T>
class Maxwell_Boltzmann
{
public:
  Maxwell_Boltzmann(T Temperatur, T mass)
    : Temperatur(Temperatur), mass(mass), k_Boltzmann(parameters::boltzmann)
    {
      const T sigma = sqrt(k_Boltzmann * Temperatur / mass);
      distribution = new Gauss_1D<T>(0.0, sigma);
     }
  ~Maxwell_Boltzmann()
  { delete distribution; } 

  T get()
    {
      return distribution->get();
    }

private:
  const T k_Boltzmann;
  const T Temperatur;
  const T mass;
  Gauss_1D<T>* distribution;
  
};

#endif

