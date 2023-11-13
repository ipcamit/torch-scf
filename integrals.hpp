

#include <memory>
class BasisFunctionSTO3G
{   
    public:
    double origin[3]= {0.0, 0.0, 0.0};
    int shell[3] = {0, 0, 0};
    double exps[3] = {0.0, 0.0, 0.0};
    double coeffs[3] = {0.0, 0.0, 0.0};
    double norm[3] = {0.0, 0.0, 0.0};

    void add_center(double * new_center);
    void normalize();

    BasisFunctionSTO3G(double * new_origin, int * new_shell, double * new_exps, double * new_coeffs);
    BasisFunctionSTO3G(){};
    void init(double * new_origin, int * new_shell, double * new_exps, double * new_coeffs);
    BasisFunctionSTO3G(BasisFunctionSTO3G &bfs);
    void copy(BasisFunctionSTO3G &bfs);
};

// basic math functions
double factorial(double n);
double factorial2(double n);
double boys(double n, double x);
double gaussian_product_center(double a, double A, double b, double B);

// one electron functions
// Hermite gaussian recursive
double E(int i, int j, int t, double Qx, double a, double b);

// gaussian overlap
double overlap(double a, int * lmn1, double A, double b, int *lmn2, double B);

// overlap matrix between two contracted gaussians
double S(BasisFunctionSTO3G * a, BasisFunctionSTO3G * b);

// kinetic energy
double kinetic(double a, int * lmn1, double * A, double b, int *lmn2, double * B);

// kinetic energy matrix
double T(BasisFunctionSTO3G * a, BasisFunctionSTO3G * b);

// coulomb auxiliary hermite integral
double R(int t, int u, int v, int n, int p, double PCx, double PCy, double PCz, double RPC);

// kinetic energy between two contracted gaussians
double nuclear_attraction(double a, int * lmn1, double * A, double b, int *lmn2, double * B, double * C);

// nuclear attraction matrix
double V(BasisFunctionSTO3G * a, BasisFunctionSTO3G * b, double * C);

// one electron integral
void get_one_electron_integrals(int n_basis, BasisFunctionSTO3G * list_of_basis, double * coords, double * S_mat, double * T_mat, double * V_mat);

// two electron repulsion integral
double electron_repulsion(double a, int * lmn1, double * A, double b, int *lmn2, double * B, double c, int * lmn3, double * C, double d, int *lmn4, double * D);

// electron repulsion between 4 contracted gaussians
double ERI(BasisFunctionSTO3G * a, BasisFunctionSTO3G * b, BasisFunctionSTO3G * c, BasisFunctionSTO3G * d);

// build two electron repulsion integrals tensor
void doERI(int n_basis, double * TwoE, BasisFunctionSTO3G * bfs);

double nuclear_nuclear_repulsion(int n_atoms, double * species, double * coords);

void get_two_electron_integrals(int n_basis, BasisFunctionSTO3G * list_of_basis, double * G_mat);

void P(int n_basis, double * C, double * P_mat);

double scf_energy(int n_basis, double * P_mat, double * S_mat, double * T_mat, double * V_mat, double * G_mat);

double scf_step(int n_atoms, int n_basis, double *C, double * species, double* coords, BasisFunctionSTO3G * basis);

// BasisFunctionSTO3G * get_list_of_basis(int n_atoms, double * coords);
BasisFunctionSTO3G * get_list_of_basis(int n_atoms, double * coords);