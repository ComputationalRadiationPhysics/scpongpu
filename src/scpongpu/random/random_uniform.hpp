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


