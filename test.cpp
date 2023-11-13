#include <iostream>
#include <cmath>
#include "gamma.hpp"

int main(){
// t.lgamma(0) = inf
// t.lgamma(1) = 0.
// t.lgamma(0.1) = 2.2527
// t.lgamma(0.01) = 4.5995
// t.lgamma(0.001) = 6.9072
// t.lgamma(10.0) = 12.8018
// t.lgamma(100.0) = 359.1342
// test a_lgamma
std::cout << "lgamma(0) = " << a_lgamma(0) << std::endl;
std::cout << "lgamma(1) = " << a_lgamma(1) << std::endl;
std::cout << "lgamma(0.1) = " << a_lgamma(0.1) << std::endl;
std::cout << "lgamma(0.01) = " << a_lgamma(0.01) << std::endl;
std::cout << "lgamma(0.001) = " << a_lgamma(0.001) << std::endl;
std::cout << "lgamma(10.0) = " << a_lgamma(10.0) << std::endl;
std::cout << "lgamma(100.0) = " << a_lgamma(100.0) << std::endl;

//gammainc(0.,0.) = nan
//gammainc(0.,0.001) = 0.0357
//gammainc(0.,0.01) = 0.1125
//gammainc(0.,0.1) = 0.3453
//gammainc(1.,0.1) = 0.0952
//gammainc(1.,0.01) = 0.0100
//gammainc(1.,0.001) = 0.0010
//gammainc(10.,0.001) = 2.7532e-37
//gammainc(10.,0.01) = 2.7308e-27
//gammainc(10.,0.1) = 2.5164e-17
//gammainc(10.,1.) = 1.1143e-07
//gammainc(10.,10.) = 0.5421

std::cout << "gamma_inc(0.5,0.) = " << gamma_inc(0.5,0.) << std::endl;
std::cout << "gamma_inc(0.5,0.001) = " << gamma_inc(0.5,0.001) << std::endl;
std::cout << "gamma_inc(0.5,0.01) = " << gamma_inc(0.5,0.01) << std::endl;
std::cout << "gamma_inc(0.5,0.1) = " << gamma_inc(0.5,0.1) << std::endl;
std::cout << "gamma_inc(1.,0.1) = " << gamma_inc(1.,0.1) << std::endl;
std::cout << "gamma_inc(1.,0.01) = " << gamma_inc(1.,0.01) << std::endl;
std::cout << "gamma_inc(1.,0.001) = " << gamma_inc(1.,0.001) << std::endl;
std::cout << "gamma_inc(10.,0.001) = " << gamma_inc(10.,0.001) << std::endl;
std::cout << "gamma_inc(10.,0.01) = " << gamma_inc(10.,0.01) << std::endl;
std::cout << "gamma_inc(10.,0.1) = " << gamma_inc(10.,0.1) << std::endl;
std::cout << "gamma_inc(10.,1.) = " << gamma_inc(10.,1.) << std::endl;
std::cout << "gamma_inc(10.,10.) = " << gamma_inc(10.,10.) << std::endl;


}