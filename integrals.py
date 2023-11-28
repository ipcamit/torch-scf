# from Joshua Goings' https://github.com/jjgoings/McMurchie-Davidson
# and myself pyhf
# Details here: https://joshuagoings.com/2017/04/28/integrals/

import torch as t
from torch.multiprocessing import Pool, Process, set_start_method
import json
from typing import List
from copy import deepcopy

try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

@t.jit.script
def fact2(n:t.Tensor):
    if n % 2==0:
        return t.lgamma(t.tensor(n/2 + 1)).exp() * 2**(n/2)
    else:
        return t.pow(2.0, (n + 1)/2) * t.lgamma(n/2 + 1).exp() / t.sqrt(t.pi)

@t.jit.script
def fact(n:t.Tensor):
    return t.lgamma(n+1).exp()

@t.jit.script
def boys(n, x):
    return t.lgamma(0.5 + n).exp() * t.special.gammainc(0.5 + n, x) / (2*x ** (0.5 + n))

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)


class BasisFunction(t.nn.Module):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  tuple of angular momentum
        exps:   list of primitive Gaussian exponents
        coeffs:  list of primitive Gaussian coefficients
        norm:   list of normalization factors for Gaussian primitives
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),exps=[],coeffs=[]):
        # self.origin = t.tensor(origin)
        # self.shell = t.tensor(shell)
        # self.exps  = t.tensor(exps)
        # self.coeffs = t.tensor(coeffs)
        # self.normalize()
        super().__init__()
        self.register_buffer('origin', t.tensor(origin))
        self.register_buffer('shell', t.tensor(shell))
        self.register_buffer('exps', t.tensor(exps))
        self.register_buffer('coeffs', t.tensor(coeffs))

    def add_center(self,center):
        self.origin = self.origin + center
        self.normalize()

    def normalize(self):
        ''' Routine to normalize the basis functions, in case they
            do not integrate to unity.
        '''
        l,m,n = self.shell
        L = l+m+n
        # self.norm is a list of length equal to number primitives
        # normalize primitives first (PGBFs)
        norm = t.sqrt(t.pow(2,2*(l+m+n)+1.5) * t.pow(self.exps,l+m+n+1.5)/
                        fact2(2*l-1)/fact2(2*m-1)/fact2(2*n-1)/t.pow(t.as_tensor(t.as_tensor(t.pi)),1.5))

        # now normalize the contracted basis functions (CGBFs)
        # Eq. 1.44 of Valeev integral whitepaper
        prefactor = t.pow(t.as_tensor(t.as_tensor(t.pi)),1.5)*\
            fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/t.pow(2.0,L)

        N = 0.0
        num_exps = len(self.exps)
        for ia in range(num_exps):
            for ib in range(num_exps):
                N += norm[ia]*norm[ib]*self.coeffs[ia]*self.coeffs[ib]/\
                         t.pow(self.exps[ia] + self.exps[ib],L+1.5)

        N *= prefactor
        N = t.pow(N,-0.5)
        for ia in range(num_exps):
            self.coeffs[ia] *= N
        self.register_buffer("norm",norm)


# read sto3g basis from json file
def read_sto3g_basis(sto3g_file):
    with open(sto3g_file, 'r') as file:
        sto3g_basis = json.load(file)
    z = int(list(sto3g_basis["elements"].keys())[0])
    number_of_exponents = len(sto3g_basis["elements"][str(z)]["electron_shells"])
    basis = sto3g_basis["elements"]["14"]["electron_shells"]
    all_basis:List[BasisFunction] = []
    for function in basis:
        for total_angular_momentum in function["angular_momentum"]:
            if total_angular_momentum == 0:
                shells = [(0.,0.,0.)]
            elif total_angular_momentum == 1:
                shells = [(1.,0.,0.), (0.,1.,0.), (0.,0.,1.)]
            elif total_angular_momentum == 2:
                shells = (2,0,0)
            coeff = list(map(float, function["coefficients"][total_angular_momentum]))
            exponents = list(map(float, function["exponents"]))
            for shell in shells:
                all_basis.append(BasisFunction(shell=shell,exps=exponents,coeffs=coeff))
    return all_basis

# print(read_sto3g_basis("Si-sto-3g.txt"))


def E(i,j,t_,Qx,a,b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t_: number nodes in Hermite (depends on type of integral, 
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    p = a + b
    q = a*b/p
    if (t_ < 0) or (t_ > (i + j)):
        # out of bounds for t_  
        return 0.0
    elif i == j == t_ == 0:
        # base case
        return t.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t_-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t_,Qx,a,b)    + \
               (t_+1)*E(i-1,j,t_+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t_-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t_,Qx,a,b)    + \
               (t_+1)*E(i,j-1,t_+1,Qx,a,b)


def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*t.pow(t.as_tensor(t.as_tensor(t.pi))/(a+b),1.5) 




def S(a,b):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, ca in enumerate(a.coeffs):
        for ib, cb in enumerate(b.coeffs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin)
    return s



# myOrigin = [1.0, 2.0, 3.0]
# myShell  = (0,0,0) # p-orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
# myExps   = [3.42525091, 0.62391373, 0.16885540] 
# myCoefs  = [0.15432897, 0.53532814, 0.44463454]
# a = BasisFunction(origin=myOrigin,shell=myShell,exps=myExps,coeffs=myCoefs)

# print(S(a,a)) # should be 1.0

# Si_basis = read_sto3g_basis("Si-sto-3g.txt")
# for i in range(len(Si_basis)):
#     for j in range(len(Si_basis)):
#         print(S(Si_basis[i],Si_basis[j]).numpy(), end=" ")
#     print()



def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*t.pow(b,2)*\
                           (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
                  m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
                  n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2




def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    t_ = 0.0
    for ia, ca in enumerate(a.coeffs):
        for ib, cb in enumerate(b.coeffs):
            t_ += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin)
    return t_




def R(t_,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals 
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function 
        PCx,y,z: Cartesian vector distance between Gaussian 
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t_ == u == v == 0:
        val += t.pow(-2*p,n)*boys(t.as_tensor(n),t.as_tensor(T))
    elif t_ == u == 0:
        if v > 1:
            val += (v-1)*R(t_,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t_,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t_ == 0:
        if u > 1:
            val += (u-1)*R(t_,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t_,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t_ > 1:
            val += (t_-1)*R(t_-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t_-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val




def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    ''' Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
               for Gaussian 'a'
         lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
         A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
         B:    list containing origin of Gaussian 'b'
         C:    list containing origin of nuclear center 'C'
    '''
    l1,m1,n1 = lmn1 
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B) # Gaussian composite center
    RPC = t.linalg.norm(P-C)

    val = 0.0
    for t_ in range(int(l1+l2+1)):
        for u in range(int(m1+m2+1)):
            for v in range(int(n1+n2+1)):
                val += E(l1,l2,t_,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t_,u,v,t.tensor(0.0),p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    val *= 2*t.tensor(t.pi)/p 
    return val


def V(a,b,C):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
       C: center of nucleus
    '''
    v = 0.0
    for ia, ca in enumerate(a.coeffs):
        for ib, cb in enumerate(b.coeffs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C)
    return v



def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
        lmn1,lmn2
        lmn3,lmn4: int tuple containing orbital angular momentum
                   for Gaussian 'a','b','c','d', respectively
        A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
    q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)
    P = gaussian_product_center(a,A,b,B) # A and B composite center
    Q = gaussian_product_center(c,C,d,D) # C and D composite center
    RPQ = t.linalg.norm(P-Q)

    val = 0.0
    for t_ in range(int(l1+l2+1)):
        for u in range(int(m1+m2+1)):
            for v in range(int(n1+n2+1)):
                for tau in range(int(l3+l4+1)):
                    for nu in range(int(m3+m4+1)):
                        for phi in range(int(n3+n4+1)):
                            val += E(l1,l2,t_,A[0]-B[0],a,b) * \
                                   E(m1,m2,u,A[1]-B[1],a,b) * \
                                   E(n1,n2,v,A[2]-B[2],a,b) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d) * \
                                   t.pow(-1,t.as_tensor(tau+nu+phi)) * \
                                   R(t_+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

    val *= 2*t.pow(t.tensor(t.tensor(t.pi)),2.5)/(p*q*t.sqrt(p+q))
    return val



def ERI(a,b,c,d):
    '''Evaluates overlap between two contracted Gaussians
        Returns float.
        Arguments:
        a: contracted Gaussian 'a', BasisFunction object
        b: contracted Gaussian 'b', BasisFunction object
        c: contracted Gaussian 'b', BasisFunction object
        d: contracted Gaussian 'b', BasisFunction object
    '''
    eri = 0.0
    for ja, ca in enumerate(a.coeffs):
        for jb, cb in enumerate(b.coeffs):
            for jc, cc in enumerate(c.coeffs):
                for jd, cd in enumerate(d.coeffs):
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                             ca*cb*cc*cd*\
                             electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                                b.exps[jb],b.shell,b.origin,\
                                                c.exps[jc],c.shell,c.origin,\
                                                d.exps[jd],d.shell,d.origin)
    return eri


def doERIs(N,TwoE,bfs):
    '''Computes the electron repulsion integrals
       Returns a 4D numpy array.
       Arguments:
       N: number of basis functions
       TwoE: 4D numpy array to store the integrals
       bfs: list of BasisFunction objects
    '''
    for i in range(int(N)):
      for j in range(i+1):
        ij = (i*(i+1)//2 + j)
        for k in range(N):
            for l in range(k+1):
                kl = (k*(k+1)//2 + l)
                if ij >= kl:
                    val = ERI(bfs[i],bfs[j],bfs[k],bfs[l])
                    TwoE[i,j,k,l] = val
                    TwoE[k,l,i,j] = val
                    TwoE[j,i,l,k] = val
                    TwoE[l,k,j,i] = val
                    TwoE[j,i,k,l] = val
                    TwoE[l,k,i,j] = val
                    TwoE[i,j,l,k] = val
                    TwoE[k,l,j,i] = val      
    return TwoE

def nuclear_nuclear_repulsion(n_atoms, coords):
    energy = t.tensor(0.0, requires_grad=True)
    zz = 14.0 * 14.0
    for i in range(n_atoms):
        for j in range(i + 1,n_atoms):
            r2 = t.sum((coords[i] - coords[j])**2)
            energy += zz/r2
    return energy

def get_one_electron_integrals(list_of_basis, coords):
    N = len(list_of_basis)
    S_mat = t.zeros((N,N))
    V_mat = t.zeros((N,N))
    T_mat = t.zeros((N,N))

    for i in range(N):
        for j in range(i + 1):
            S_mat[i, j] = S(list_of_basis[i], list_of_basis[j])
            S_mat[j, i] = S_mat[i, j]
            T_mat[i, j] = T(list_of_basis[i], list_of_basis[j])
            T_mat[j, i] = T_mat[i, j]
            for atom in coords:
                V_mat[i, j] += -14 * V(list_of_basis[i], list_of_basis[j], atom) # TODO put species
                V_mat[j, i] = V_mat[i, j]
    return S_mat, T_mat, V_mat

def get_two_electron_integrals(list_of_basis):
    N = len(list_of_basis)
    G_mat = t.zeros((N,N,N,N))
    G_mat = doERIs(N, G_mat, list_of_basis)
    return G_mat

def scf_2_iterations(learned_C, S_mat, T_mat, V_mat, G_mat):
    P = t.zeros_like(S_mat)
    for i in range(len(learned_C)):
        for j in range(len(learned_C)):
            P[i, j] = learned_C[i] * learned_C[j]

    G = t.zeros_like(S_mat)
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            for x in range(P.shape[0]):
                for y in range(P.shape[0]):
                    G[i, j] += P[x, y]*(G_mat[i, j, y, x] - 0.5*G_mat[i, x, y, j])

    H = T_mat + V_mat
    F = H + G
    s, U = t.linalg.eig(S_mat)
    X = U @ t.diag(t.sqrt(t.diag(1/s))) @ U.T
    Fp = X.T @ F @ X
    Cp, Cp_vec = t.linalg.eig(Fp)
    C = X @ Cp_vec
    energy = 0.0

    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            P[i, j] += C[i] * C[j] # first correction

    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            energy += P[i, j]*(H[i, j] + F[i, j]) # 2H + G

    return energy

def scf_energy(learned_C, S_mat, T_mat, V_mat, G_mat):
    P = t.zeros_like(S_mat)
    for i in range(len(learned_C)):
        for j in range(len(learned_C)):
            P[i, j] = learned_C[i] * learned_C[j]

    G = t.zeros_like(S_mat)
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            for x in range(P.shape[0]):
                for y in range(P.shape[0]):
                    G[i, j] += P[x, y]*(G_mat[i, j, y, x] - 0.5*G_mat[i, x, y, j])

    H = T_mat + V_mat
    F = 2 * H + G
    energy = (P*F).sum()
    return energy

def scf_step(learned_C, basis_list, coords):
    S_mat, T_mat, V_mat = get_one_electron_integrals(basis_list, coords)
    G_mat = get_two_electron_integrals(basis_list)
    return scf_energy(learned_C, S_mat, T_mat, V_mat, G_mat)

def scf_step_2(learned_C, basis_list, coords):
    S_mat, T_mat, V_mat = get_one_electron_integrals(basis_list, coords)
    G_mat = get_two_electron_integrals(basis_list)
    return scf_2_iterations(learned_C, S_mat, T_mat, V_mat, G_mat)

def get_basis_list(coords):
    n_atoms = coords.shape[0]
    basis_Si = read_sto3g_basis("Si-sto-3g.txt")
    basis_list = []
    for i in range(n_atoms):
        for basis in basis_Si:
            new_centered_basis = deepcopy(basis)
            new_centered_basis.add_center(coords[i])
            basis_list.append(new_centered_basis)
    return basis_list



def scf_step_simplified(learned_C, coords):
    basis_list = get_basis_list(coords)
    S_mat, T_mat, V_mat = get_one_electron_integrals(basis_list, coords)
    print("Finished One e integrals")
    G_mat = get_two_electron_integrals(basis_list)
    print("Finished two e integrals")
    return scf_energy(learned_C, S_mat, T_mat, V_mat, G_mat)
