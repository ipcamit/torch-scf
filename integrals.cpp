// C++ version of one and two electron integrals
// McMurchie-Davidson recursion

#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "integrals.hpp"
#include "gamma.hpp"

#define M_SQRT_PI  1.7724538509055160272981674833

int enzyme_dup, enzyme_const;


// Factorial function
double factorial(double n){
    return std::exp(a_lgamma(n+1));
}

double factorial2(double n){
    // even n
    if (static_cast<int>(n) % 2 == 0){
        return std::exp(a_lgamma(n/2 + 1)) * std::pow(2,n/2);
    } else {
        return std::exp(a_lgamma((n+1)/2)) * std::pow(2,(n+1)/2) / M_SQRT_PI;
    }
}

double boys(double n, double x){
    return std::exp(a_lgamma(n+0.5)) * gamma_inc(0.5 + n, x) / std::pow(2.0*x, n+0.5);
}

double gaussian_product_center(double a, double A, double b, double B) {
    return (a*A + b*B) / (a + b);
}

void BasisFunctionSTO3G::init(double* new_origin, int* new_shell, double* new_exps, double* new_coeffs) {
    for (int i = 0; i < 3; i++){
        origin[i] = new_origin[i];
        shell[i] = new_shell[i];
        exps[i] = new_exps[i];
        coeffs[i] = new_coeffs[i];
    }
    norm[0] = 0.0;
    norm[1] = 0.0;
    norm[2] = 0.0;
}

BasisFunctionSTO3G::BasisFunctionSTO3G(double* new_origin, int* new_shell, double* new_exps, double* new_coeffs) {
    init(new_origin, new_shell, new_exps, new_coeffs);
}

void BasisFunctionSTO3G::add_center(double* new_center){
    for (int i = 0; i < 3; i++){
        origin[i] += new_center[i];
    }
}

BasisFunctionSTO3G::BasisFunctionSTO3G(BasisFunctionSTO3G & bfs){
    init(bfs.origin, bfs.shell, bfs.exps, bfs.coeffs);
}

void BasisFunctionSTO3G::copy(BasisFunctionSTO3G & bfs){
    init(bfs.origin, bfs.shell, bfs.exps, bfs.coeffs);
}

void BasisFunctionSTO3G::normalize(){
        const int l = shell[0];
        const int m = shell[1];
        const int n = shell[2];
        int L = l+m+n;
        double lmn_pow = std::pow(2,2*(l+m+n)+1.5);
        double fact2_1 = factorial2(static_cast<double>(2*l-1));
        double fact2_2 = factorial2(static_cast<double>(2*m-1));
        double fact2_3 = factorial2(static_cast<double>(2*n-1)); 
        double M_PI_1_5 = std::pow(M_PI,1.5);

        norm[0] = lmn_pow * std::pow(exps[0],L+1.5)/fact2_1/fact2_2/fact2_3/M_PI_1_5;
        norm[1] = lmn_pow * std::pow(exps[1],L+1.5)/fact2_1/fact2_2/fact2_3/M_PI_1_5;
        norm[2] = lmn_pow * std::pow(exps[2],L+1.5)/fact2_1/fact2_2/fact2_3/M_PI_1_5;

        double prefactor = M_PI_1_5*fact2_1*fact2_2*fact2_3/std::pow(2.0,L);
        double N = 0.0;
        int num_exps = 3;
        for (int ia = 0; ia < num_exps; ia++){
            for (int ib = 0; ib < num_exps; ib++){
                N += norm[ia]*norm[ib]*coeffs[ia]*coeffs[ib]/
                         std::pow(exps[ia] + exps[ib],L+1.5);
            }
        }

        N *= prefactor;
        N = std::pow(N,-0.5);
        for (int ia = 0; ia < num_exps; ia++){
            coeffs[ia] *= N;
        }
}

// Hermite gaussian recursive
double E(int i, int j, int t, double Qx, double a, double b){
    double p = a + b;
    double q = a*b/p;
    if ((t < 0) || (t > (i + j))){
        return 0.0;
    } else if ((i == 0) && (j == 0) && (t == 0)){
        return std::exp(-q*Qx*Qx);
    } else if (j == 0){
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b);
    } else {
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b);
    }
}

// gaussian overlap
double overlap(double a, int* lmn1, double* A, double b, int* lmn2, double* B){
    double S1 = E(lmn1[0],lmn2[0],0,A[0]-B[0],a,b);
    double S2 = E(lmn1[1],lmn2[1],0,A[1]-B[1],a,b);
    double S3 = E(lmn1[2],lmn2[2],0,A[2]-B[2],a,b);
    return S1*S2*S3*std::pow(M_PI/(a+b),1.5);
}

double S(BasisFunctionSTO3G *a, BasisFunctionSTO3G *b){
    double s = 0.0;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            s += a->norm[i]*b->norm[j]*a->coeffs[i]*b->coeffs[j]*\
                     overlap(a->exps[i],a->shell,a->origin,
                     b->exps[j],b->shell,b->origin);
        }
    }
    return s;
}

double kinetic(double a, int *lmn1, double *A, double b, int *lmn2, double *B){
    int l1 = lmn1[0];
    int m1 = lmn1[1];
    int n1 = lmn1[2];
    int l2 = lmn2[0];
    int m2 = lmn2[1];
    int n2 = lmn2[2];

    double term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,lmn1,A,b,lmn2,B);
    
    int lmn_arg_1[3] = {l2+2,m2,n2};
    int lmn_arg_2[3] = {l2,m2+2,n2};
    int lmn_arg_3[3] = {l2,m2,n2+2};
    double term1 = -2*std::pow(b,2)*\
                            (overlap(a,lmn1,A,b,lmn_arg_1,B) +
                             overlap(a,lmn1,A,b,lmn_arg_2,B) +
                             overlap(a,lmn1,A,b,lmn_arg_3,B));
    
    lmn_arg_1[0] = l2-2;
    lmn_arg_2[1] = m2-2;
    lmn_arg_3[2] = n2-2;
    double term2 = -0.5*(l2*(l2-1)*overlap(a,lmn1,A,b,lmn_arg_1,B) +
                         m2*(m2-1)*overlap(a,lmn1,A,b,lmn_arg_2,B) +
                         n2*(n2-1)*overlap(a,lmn1,A,b,lmn_arg_3,B));
    return term0+term1+term2;
}

double T(BasisFunctionSTO3G *a, BasisFunctionSTO3G *b){
    double t = 0.0;
    for (int ia = 0; ia < 3; ia++){
        for (int ib = 0; ib < 3; ib++){
            t += a->norm[ia]*b->norm[ib]*a->coeffs[ia]*b->coeffs[ib]*\
                     kinetic(a->exps[ia],a->shell,a->origin,
                     b->exps[ib],b->shell,b->origin);
        }
    }
    return t;
}

double R(int t, int u, int v, int n, int p, double PCx, double PCy, double PCz, double RPC){
    double T = p*RPC*RPC;
    double val = 0.0;
    if (t == 0 && u == 0 && v == 0){
        val += std::pow(-2 * p, n) * boys(n, T);
    } else if (t == 0 && u == 0){
        if (v > 1){
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC);
        }
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC);
    } else if (t == 0){
        if (u > 1){
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC);
        }
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC);
    } else {
        if (t > 1){
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC);
        }
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC);
    }
    return val;
}

double nuclear_attraction(double a, int *lmn1, double *A, double b, int *lmn2, double *B, double *C){
    int l1 = lmn1[0];
    int m1 = lmn1[1];
    int n1 = lmn1[2];
    int l2 = lmn2[0];
    int m2 = lmn2[1];
    int n2 = lmn2[2];
    double p = a + b;
    double P[3];
    P[0] = gaussian_product_center(a,A[0],b,B[0]);
    P[1] = gaussian_product_center(a,A[1],b,B[1]);
    P[2] = gaussian_product_center(a,A[2],b,B[2]);
    double RPC = std::sqrt(std::pow(P[0]-C[0],2) + std::pow(P[1]-C[1],2) + std::pow(P[2]-C[2],2));
    double val = 0.0;
    for (int t = 0; t < l1+l2+1; t++){
        for (int u = 0; u < m1+m2+1; u++){
            for (int v = 0; v < n1+n2+1; v++){
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC);
            }
        }
    }
    val *= 2*M_PI/p;
    return val;
}

double V(BasisFunctionSTO3G *a, BasisFunctionSTO3G *b, double *C){
    double v = 0.0;
    for (int ia = 0; ia < 3; ia++){
        for (int ib = 0; ib < 3; ib++){
            v += a->norm[ia]*b->norm[ib]*a->coeffs[ia]*b->coeffs[ib]*\
                     nuclear_attraction(a->exps[ia],a->shell,a->origin,
                     b->exps[ib],b->shell,b->origin,C);
        }
    }
    return v;
}

void get_one_electron_integrals(int n_gaussian, BasisFunctionSTO3G *list_of_basis, double *coords, double * S_mat, double * T_mat, double * V_mat){
    for (int i = 0; i < n_gaussian; i++){
        for (int j = 0; j < i+1; j++){
            S_mat[i*n_gaussian+j] = S(&list_of_basis[i], &list_of_basis[j]);
            S_mat[j*n_gaussian+i] = S_mat[i*n_gaussian+j];
            T_mat[i*n_gaussian+j] = T(&list_of_basis[i], &list_of_basis[j]);
            T_mat[j*n_gaussian+i] = T_mat[i*n_gaussian+j];
            for (int atom = 0; atom < 3; atom++){
                V_mat[i*n_gaussian+j] += -14 * V(&list_of_basis[i], &list_of_basis[j], &coords[atom*3]);
                V_mat[j*n_gaussian+i] = V_mat[i*n_gaussian+j];
            }
        }
    }
}

// two electron integrals
double electron_repulsion(double a, int * lmn1, double * A, double b, int *lmn2, double * B, double c, int * lmn3, double * C, double d, int *lmn4, double * D){
    int l1 = lmn1[0];
    int m1 = lmn1[1];
    int n1 = lmn1[2];
    int l2 = lmn2[0];
    int m2 = lmn2[1];
    int n2 = lmn2[2];
    int l3 = lmn3[0];
    int m3 = lmn3[1];
    int n3 = lmn3[2];
    int l4 = lmn4[0];
    int m4 = lmn4[1];
    int n4 = lmn4[2];
    double p = a+b;
    double q = c+d;
    double alpha = p*q/(p+q);
    double P[3];
    double Q[3];
    P[0] = gaussian_product_center(a,A[0],b,B[0]);
    P[1] = gaussian_product_center(a,A[1],b,B[1]);
    P[2] = gaussian_product_center(a,A[2],b,B[2]);
    Q[0] = gaussian_product_center(c,C[0],d,D[0]);
    Q[1] = gaussian_product_center(c,C[1],d,D[1]);
    Q[2] = gaussian_product_center(c,C[2],d,D[2]);
    double RPQ = sqrt(pow(P[0]-Q[0],2)+pow(P[1]-Q[1],2)+pow(P[2]-Q[2],2));
    double val = 0.0;
    for (int t = 0; t < l1+l2+1; t++){
        for (int u = 0; u < m1+m2+1; u++){
            for (int v = 0; v < n1+n2+1; v++){
                for (int tau = 0; tau < l3+l4+1; tau++){
                    for (int nu = 0; nu < m3+m4+1; nu++){
                        for (int phi = 0; phi < n3+n4+1; phi++){
                            val += E(l1,l2,t,A[0]-B[0],a,b) * \
                                   E(m1,m2,u,A[1]-B[1],a,b) * \
                                   E(n1,n2,v,A[2]-B[2],a,b) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d) * \
                                   pow(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ);
                        }
                    }
                }
            }
        }
    }
    val *= 2*pow(M_PI,2.5)/(p*q*sqrt(p+q));
    return val;
}


double ERI(BasisFunctionSTO3G * a, BasisFunctionSTO3G * b, BasisFunctionSTO3G * c, BasisFunctionSTO3G * d){
    double eri = 0.0;
    for (int ja = 0; ja < 3; ja++){
        for (int jb = 0; jb < 3; jb++){
            for (int jc = 0; jc < 3; jc++){
                for (int jd = 0; jd < 3; jd++){
                    eri += a->norm[ja]*b->norm[jb]*c->norm[jc]*d->norm[jd]*\
                             a->coeffs[ja]*b->coeffs[jb]*c->coeffs[jc]*d->coeffs[jd]*\
                             electron_repulsion(a->exps[ja],a->shell,a->origin,\
                                                b->exps[jb],b->shell,b->origin,\
                                                c->exps[jc],c->shell,c->origin,\
                                                d->exps[jd],d->shell,d->origin);
                }
            }
        }
    }
    return eri;
}

void doERI(int n_basis, double * TwoE, BasisFunctionSTO3G * bfs){
    int N3 = n_basis*n_basis*n_basis;
    int N2 = n_basis*n_basis;
    for (int i = 0; i < n_basis; i++){
        for (int j = 0; j < i+1; j++){
            int ij = (i*(i+1)/2 + j);
            for (int k = 0; k < n_basis; k++){
                for (int l = 0; l < k+1; l++){
                    int kl = (k*(k+1)/2 + l);
                    if (ij >= kl){
                        double val = ERI(&bfs[i],&bfs[j],&bfs[k],&bfs[l]);
                        TwoE[i*N3+j*N2+k*n_basis+l] = val;
                        TwoE[k*N3+l*N2+i*n_basis+j] = val;
                        TwoE[j*N3+i*N2+l*n_basis+k] = val;
                        TwoE[l*N3+k*N2+j*n_basis+i] = val;
                        TwoE[j*N3+i*N2+k*n_basis+l] = val;
                        TwoE[l*N3+k*N2+i*n_basis+j] = val;
                        TwoE[i*N3+j*N2+l*n_basis+k] = val;
                        TwoE[k*N3+l*N2+j*n_basis+i] = val;
                    }
                }
            }
        }
    }
}

double nuclear_nuclear_repulsion(int n_atoms, double * species, double * coords){
    double val = 0.0;
    for (int i = 0; i < n_atoms; i++){
        for (int j = i+1; j < n_atoms; j++){
            double r = sqrt(pow(coords[i*3+0]-coords[j*3+0],2)+\
                            pow(coords[i*3+1]-coords[j*3+1],2)+\
                            pow(coords[i*3+2]-coords[j*3+2],2));
            val += species[i]*species[j]/r;
        }
    }
    return val;
}

void get_two_electron_integrals(int n_basis, BasisFunctionSTO3G * list_of_basis, double * G_mat){
    doERI(n_basis,G_mat, list_of_basis);
}

void P(int n_basis, double *C, double * P_mat){
    for (int i = 0; i < n_basis; i++){
        for (int j = i; j < n_basis; j++){
            P_mat[i*n_basis+j] = C[i]*C[j];
            P_mat[j*n_basis+i] = P_mat[i*n_basis+j];
        }
    }
}

double scf_energy(int n_basis, double * P_mat, double * S_mat, double * T_mat, double * V_mat, double * G_mat){
    int N3 = n_basis*n_basis*n_basis;
    int N2 = n_basis*n_basis;

    double * G = new double[n_basis*n_basis];
    for (int i = 0; i < n_basis; i++){
        for (int j = 0; j < n_basis; j++){
            G[i*n_basis+j] = 0.0;
            for (int x = 0; x < n_basis; x++){
                for (int y = 0; y < n_basis; y++){
                    G[i*n_basis+j] += P_mat[x*n_basis+y]*(G_mat[i*N3+j*N2+y*n_basis+x] - 0.5*G_mat[i*N3+x*N2+y*n_basis+j]);
                }
            }
        }
    }

    double energy = 0.0;
    for (int i = 0; i < n_basis; i++){
        for (int j = 0; j < n_basis; j++){
            energy += P_mat[i*n_basis+j]*(2*(T_mat[i*n_basis+j] + V_mat[i*n_basis+j]) + G[i*n_basis+j]);
        }
    }

    delete [] G;
    return energy;
}

double scf_step(int n_atoms, int n_basis_per_atom, double *C, double * species, double* coords, BasisFunctionSTO3G *list_of_basis){
    int n_basis = n_atoms*n_basis_per_atom;
    double * S_mat = new double[n_basis*n_basis];
    double * T_mat = new double[n_basis*n_basis];
    double * V_mat = new double[n_basis*n_basis];
    double * G_mat = new double[n_basis*n_basis*n_basis*n_basis];
    double * P_mat = new double[n_basis*n_basis];

    get_one_electron_integrals(n_basis, list_of_basis, coords, S_mat, T_mat, V_mat);
    get_two_electron_integrals(n_basis, list_of_basis, G_mat);
    double energy = scf_energy(n_basis, P_mat, S_mat, T_mat, V_mat, G_mat);

    delete [] S_mat;
    delete [] T_mat;
    delete [] V_mat;
    delete [] G_mat;
    delete [] P_mat;
    return energy;
}

BasisFunctionSTO3G * get_list_of_basis(int n_atoms, double * coords){
    // setup file io later
    BasisFunctionSTO3G Si_basis_1s, Si_basis_2s, Si_basis_2p, Si_basis_3s, Si_basis_3p;
    Si_basis_1s.coeffs[0] = 0.1543289673E+00;
    Si_basis_1s.coeffs[1] = 0.5353281423E+00;
    Si_basis_1s.coeffs[2] = 0.4446345422E+00;

    Si_basis_2s.coeffs[0] = -0.9996722919E-01;
    Si_basis_2s.coeffs[1] = 0.3995128261E+00;
    Si_basis_2s.coeffs[2] = 0.7001154689E+00;

    Si_basis_2p.coeffs[0] = 0.1559162750E+00;
    Si_basis_2p.coeffs[1] = 0.6076837186E+00;
    Si_basis_2p.coeffs[2] = 0.3919573931E+00;

    Si_basis_3s.coeffs[0] = -0.2196203690E+00;
    Si_basis_3s.coeffs[1] = 0.2255954336E+00;
    Si_basis_3s.coeffs[2] = 0.9003984260E+00;

    Si_basis_3p.coeffs[0] = 0.1058760429E-01;
    Si_basis_3p.coeffs[1] = 0.5951670053E+00;
    Si_basis_3p.coeffs[2] = 0.4620010120E+00;

    // --------------------------------------
    Si_basis_1s.exps[0] = 0.4077975514E+03;
    Si_basis_1s.exps[1] = 0.7428083305E+02;
    Si_basis_1s.exps[2] = 0.2010329229E+02;

    Si_basis_2s.exps[0] = 0.2319365606E+02;
    Si_basis_2s.exps[1] = 0.5389706871E+01;
    Si_basis_2s.exps[2] = 0.1752899952E+01;

    Si_basis_2p.exps[0] = 0.2319365606E+02;
    Si_basis_2p.exps[1] = 0.5389706871E+01;
    Si_basis_2p.exps[2] = 0.1752899952E+01;

    Si_basis_3s.exps[0] = 0.1478740622E+01;
    Si_basis_3s.exps[1] = 0.4125648801E+00;
    Si_basis_3s.exps[2] = 0.1614750979E+00;
    
    Si_basis_3p.exps[0] = 0.1478740622E+01;
    Si_basis_3p.exps[1] = 0.4125648801E+00;
    Si_basis_3p.exps[2] = 0.1614750979E+00;

    // --------------------------------------

    int shells[9][3] = {0, 0, 0,  // Si 1s
                      0, 0, 0,  // Si 2s
                      1, 0, 0,  // Si 2px
                      0, 1, 0,  // Si 2py
                      0, 0, 1,  // Si 2pz
                      0, 0, 0,  // Si 3s
                      1, 0, 0,  // Si 3px
                      0, 1, 0,  // Si 3py
                      0, 0, 1}; // Si 3pz
    int n_shells = 9;

    // BasisFunctionSTO3G * list_of_basis = new BasisFunctionSTO3G[n_atoms * n_shells];
    auto list_of_basis = new  BasisFunctionSTO3G[n_atoms * n_shells];
    
    for (int i = 0; i < n_atoms; i++){
        list_of_basis[i*n_shells + 0].copy(Si_basis_1s);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+0].origin);
        std::copy_n( shells[0], 3, list_of_basis[i*n_shells+0].shell);
        list_of_basis[i*n_shells+0].normalize();
        list_of_basis[i*n_shells + 1].copy(Si_basis_2s);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+1].origin);
        std::copy_n( shells[1], 3, list_of_basis[i*n_shells+1].shell);
        list_of_basis[i*n_shells+1].normalize();
        list_of_basis[i*n_shells + 2].copy(Si_basis_2p);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+2].origin);
        std::copy_n( shells[2], 3, list_of_basis[i*n_shells+2].shell);
        list_of_basis[i*n_shells+2].normalize();
        list_of_basis[i*n_shells + 3].copy(Si_basis_2p);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+3].origin);
        std::copy_n( shells[3], 3, list_of_basis[i*n_shells+3].shell);
        list_of_basis[i*n_shells+3].normalize();
        list_of_basis[i*n_shells + 4].copy(Si_basis_2p);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+4].origin);
        std::copy_n( shells[4], 3, list_of_basis[i*n_shells+4].shell);
        list_of_basis[i*n_shells+4].normalize();
        list_of_basis[i*n_shells + 5].copy(Si_basis_3s);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+5].origin);
        std::copy_n( shells[5], 3, list_of_basis[i*n_shells+5].shell);
        list_of_basis[i*n_shells+5].normalize();
        list_of_basis[i*n_shells + 6].copy(Si_basis_3p);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+6].origin);
        std::copy_n( shells[6], 3, list_of_basis[i*n_shells+6].shell);
        list_of_basis[i*n_shells+6].normalize();
        list_of_basis[i*n_shells + 7].copy(Si_basis_3p);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+7].origin);
        std::copy_n( shells[7], 3, list_of_basis[i*n_shells+7].shell);
        list_of_basis[i*n_shells+7].normalize();
        list_of_basis[i*n_shells + 8].copy(Si_basis_3p);
        std::copy_n( coords + 3*i, 3, list_of_basis[i*n_shells+8].origin);
        std::copy_n( shells[8], 3, list_of_basis[i*n_shells+8].shell);
        list_of_basis[i*n_shells+8].normalize();
    }
    return list_of_basis;
}

// double scf_step(int n_atoms, int n_basis_per_atom, double *C, double * species, double* coords, BasisFunctionSTO3G *list_of_basis)
double __enzyme_autodiff(double (*) (int, int, double *, double *, double *, BasisFunctionSTO3G *),
                        int, int,
                        int, int,
                        int, double *, double *,
                        int, double *,
                        int, double *, double *,
                        int, BasisFunctionSTO3G *);

void grad_scf_step(int n_atoms, int n_basis_per_atom, double *C,double *dC, double * species, double* coords, double * d_coords, BasisFunctionSTO3G *list_of_basis){
    __enzyme_autodiff(scf_step, 
                    enzyme_const, n_atoms, enzyme_const, n_atoms, 
                    enzyme_dup, C, dC, 
                    enzyme_const, species, 
                    enzyme_dup, coords, d_coords, 
                    enzyme_const, list_of_basis);
}

namespace py = pybind11;

PYBIND11_MODULE(scf, m){
    m.def("scf_step", [](int n_atoms, py::array_t<double> C, py::array_t<double> species, py::array_t<double> coords){
        auto C_buf = C.request();
        auto species_buf = species.request();
        auto coords_buf = coords.request();
        int n_basis_per_atom = 9;
        double *C_ptr = (double *) C_buf.ptr;
        double *species_ptr = (double *) species_buf.ptr;
        double *coords_ptr = (double *) coords_buf.ptr;
        auto list_of_basis = get_list_of_basis(n_atoms,coords_ptr);

        return scf_step(n_atoms, n_basis_per_atom, C_ptr, species_ptr, coords_ptr, list_of_basis);
    });
}
