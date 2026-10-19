
#include <ontheFly.hpp>
#include <fstream>   // For file handling
#include <algorithm>
#include <stdexcept>
#include <string>
namespace cie
{
namespace otFly {

ontheFly::ontheFly(int nsteps, int ndofglobal):
                    ndofglobal_(ndofglobal),
                    nsteps_(nsteps),
                    u_(nsteps + 1, ndofglobal, 0.0),
                    fKu_invM_(ndofglobal, 0.0),
                    fku_(ndofglobal,0.0),
                    Mu_(ndofglobal,0.0),
                    up1_(ndofglobal,0.0),
                    up2_(ndofglobal,0.0)

                    {    }//constructor1

ontheFly::ontheFly(int nsteps, int ndofglobal, std::string mode):
                    ndofglobal_(ndofglobal),
                    nsteps_(nsteps)
    {    
        // Matrix mode_u;
        if (mode=="storage"){
            // this mode does not save timesteps. only the necessary timesteps
            // for calculations.
            Matrix mode_u(1, ndofglobal, 0.0);
            u_=mode_u;
            std::vector<double> v(ndofglobal, 0.0);
            fKu_invM_ = v;
            fku_      = v;
            Mu_       = v;
            up1_      = v;
            up2_      = v;
        }
        else if (mode=="strg_1"){
            //strg_1 is the first mode of storage? 
            // TODO: be more specific with the naming
            Matrix mode_u(1, ndofglobal, 1.0);
            u_=mode_u;
            std::vector<double> v(ndofglobal, 0.0);
            fKu_invM_ = v;
            fku_      = v;
            Mu_       = v;
            up1_      = v;
            up2_      = v;
        }
        else if (mode=="strg_cuda"){
            Matrix mode_u(1, ndofglobal, 1.0);
            u_=mode_u;
            //not initializing any of the other variables.
        }
        else {
            Matrix mode_u(nsteps + 1, ndofglobal, 0.0);
            u_=mode_u;
            std::vector<double> v(ndofglobal, 0.0);
            fKu_invM_ = v;
            fku_      = v;
            Mu_       = v;
            up1_      = v;
            up2_      = v;
        }

    }//constructor2

void ontheFly::updated_us(){
    std::copy(up1_.begin(), up1_.end(), up2_.begin());

    for (size_t i=0;i<ndofglobal_;i++){
        up1_[i] =  u_(0,i);
    }
    //helpful for debugging
    // auto max_it = std::max_element(up1_.begin(), up1_.end());
    // std::cout<<"success in copying matrix to vector Max:"<<
    // *max_it<<std::endl;
}


Matrix ontheFly::otF_Solve(const Params& params, bool debug) {
    compute_Mass_Cvector_OMP(Mu_,params);

    for (size_t it = 2; it <= nsteps_; ++it) {
        compute_invM_fKu(params, it);
        for (size_t r = 0; r < ndofglobal_; ++r) {
            u_(it, r) = 2.0 * u_(it - 1, r) - u_(it - 2, r) +fKu_invM_[r];//+ compute_invM_fKu(params, it)[r];
        }
        if (debug) {
            if (it % (params.samplingFrecuency()) == 0) {
                std::cout << "Iteration:\t" << it;
                std::cout << "\t with u_(" << ndofglobal_ / 2;
                std::cout << ")= " << u_(it, ndofglobal_ / 2) << std::endl;
            }
        }
    }
    return u_;
}
Matrix ontheFly::otF_Solve_OMP(const Params& params, bool debug) {
    /*
    OpenMP solver that keeps the whole solution in memory: u_ holds all
    (nsteps+1) x ndof values (9.9 GB at 80x80, p = 4, 12,000 steps), so use
    it only for small runs. Reached by params.solve("ontheFly_omp"); u_ is
    returned to Python (pybind) as a NumPy array of the same shape.

    Difference to otF_Solve_OMP_write: the update reads u^n and u^(n-1)
    straight from rows it-1 and it-2 of u_. up1_ is needed only because
    compute_fKu_Cvector_OMP takes the field as a vector, so after each step
    up1_ = u_(it, :) is the u^n of the next step's K u; up2_ is not used.
    */
    std::cout<<"Starting OMP solver..."<<std::endl;
    //initializing Mu
    compute_Mass_Cvector_OMP(Mu_,params);

    for (size_t it = 2; it <= nsteps_; ++it) {
        compute_invM_fKu_OMP(params, it);
        for (size_t r = 0; r < ndofglobal_; ++r) {
            u_(it, r) = 2.0 * u_(it - 1, r) - u_(it - 2, r) +fKu_invM_[r];
            up1_[r] = u_(it, r);          // u^n for the next step's K u
        }
        if (debug) {
            if (it % (params.samplingFrecuency()) == 0) {
                std::cout << "Iteration:\t" << it;
                std::cout << "\t with u_(" << ndofglobal_ / 2;
                std::cout << ")= " << u_(it, ndofglobal_ / 2) << std::endl;
            }
        }
    }
    return u_;
}
void ontheFly::otF_Solve_OMP_write(const Params& params, std::string fname,
     bool write, bool debug) {

    std::stringstream debug_buffer;

    debug_buffer<<"Starting OMP solver to write data into: "<<
    fname<<std::endl;
    compute_Mass_Cvector_OMP(Mu_,params);

    for (int it = 2; it <= nsteps_; ++it) {

        compute_invM_fKu_OMP(params, it);
        #pragma omp parallel for
        for (size_t r = 0; r < ndofglobal_; ++r) {
            u_(0, r) = 2.0 * up1_[r] - up2_[r] +fKu_invM_[r];
        }
        this->updated_us();
        if (it % (params.samplingFrecuency()) == 0) {
        // if (it % 1 == 0) {
            if (write){
                debug_buffer << "Writing:\t" << it;
                writeU_inter(u_, fname);
            }
            if (debug) {
                debug_buffer<< "\nIteration:\t" << it
                << "\t with u_(" << ndofglobal_ / 2
                << ")= " << u_(0, ndofglobal_ / 2) 
                << std::endl;
            }
        }
    }
    if (debug) std::cout<<debug_buffer.str();
}


void ontheFly::compute_invM_fKu(const Params& params, const size_t it) {
    std::vector<double> fku = compute_fKu_Cvector(params, it);
    std::vector<double> m = compute_Mass_Cvector(params);

    for (size_t i = 0; i < fku.size(); ++i) {
        fKu_invM_[i] = fku[i] / m[i];

    }
}
// void ontheFly::compute_invM_fKuCAT(const Params& params, const double it) {
    
//     compute_fKu_Cvector_OMPxCAT(fku_,u_,params, it);
//     // compute_Mass_Cvector_OMP(Mu_,params);

//     #pragma omp parallel for
//     for (size_t i = 0; i < fku_.size(); ++i) {
//         fKu_invM_[i] = fku_[i] / Mu_[i];
// }
// }



void ontheFly::compute_invM_fKu_OMP(const Params& params, const size_t it) {

    compute_fKu_Cvector_OMP(fku_,up1_,params, it);
    // compute_Mass_Cvector_OMP(Mu_,params);

    #pragma omp parallel for
    for (size_t i = 0; i < fku_.size(); ++i) {
        fKu_invM_[i] = fku_[i] / Mu_[i];
    }
}

std::vector<double> ontheFly::compute_fKu_Cvector(const Params& params,
                                                        const size_t it) {
    std::vector<double> fKu(params.ndofglobal(), 0.0);

    double ft_dt = params.ft(it * params.dt());

    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {

            std::vector<double> locationMap = params.createLocationMap(ielement, jelement);

            std::vector<double> u_m = slice(it-1, locationMap);

            std::vector<double> Fi_element = alg_84(2, params.B(), params.Bdiff(), params.M(), u_m);

            std::vector<double> Fe_element = params.get_Force_xy(ielement, jelement);

            alg_82(2, params.B(), params.G(), Fe_element);


            for (size_t i = 0; i < locationMap.size(); ++i) {
                fKu[locationMap[i]] += std::pow(params.dt(), 2) * (Fe_element[i] * ft_dt - Fi_element[i]);
            }
        }
    }
    return fKu;
}

void compute_fKu_Cvector_OMP(std::vector<double>& fku,
    const std::vector<double>& u_p1,const Params& params, const double it) {
    std::fill(fku.begin(),fku.end(),0.0);
    double ft_dt = params.ft(it * params.dt());

    #pragma omp parallel for collapse (2)
    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {
            std::vector<double> locationMap = params.createLocationMap(ielement, jelement);
            std::vector<double> u_m = slice(u_p1, locationMap);
            std::vector<double> Fi_element = alg_84(2, params.B(), params.Bdiff(), params.M(), u_m);
            std::vector<double> Fe_element = params.get_Force_xy(ielement, jelement);
            alg_82(2, params.B(), params.G(), Fe_element);
            for (size_t i = 0; i < locationMap.size(); ++i) {
                #pragma omp atomic
                fku[locationMap[i]] += std::pow(params.dt(), 2) * (Fe_element[i] * ft_dt - Fi_element[i]);
            }
        }
    }
}


std::vector<double> ontheFly::compute_Mass_Cvector(const Params& params) {

    std::vector<double> Mu(params.ndofglobal(),0.0);

    std::vector<double> u(params.ndofelement(),1.0);

    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {

            std::vector<double> coordinates = params.gllcoords();

            std::vector<double> Me_element(params.ndofelement(),1.0);
            alg_82(2, params.B(), params.G(), Me_element);

            std::vector<double> locationMap = params.createLocationMap(ielement, jelement);

            for (size_t i = 0; i < locationMap.size(); ++i) {
                Mu[locationMap[i]] += Me_element[i];
            }
        }
    }
    return Mu;
}
void compute_Mass_Cvector_OMP(std::vector<double>& mu,const Params& params) {

    std::fill(mu.begin(),mu.end(),0.0);

    std::vector<double> u(params.ndofelement(),1.0);

    #pragma omp parallel for collapse (2)
    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {

            // std::vector<double> coordinates = params.gllcoords();
            std::vector<double> M_el(params.ndofelement(),1.0);
        
            alg_82(2, params.B(), params.G(), M_el);

            std::vector<double> locationMap = 
                    params.createLocationMap(ielement, jelement);

            for (size_t i = 0; i < locationMap.size(); ++i) {
                #pragma omp atomic
                mu[locationMap[i]] += M_el[i];
            }
        }
    }
}

double ontheFly::operator( )(size_t i, size_t j) const {
    runtime_check(i < u_.size1(), "Index i out of range.");
    runtime_check(j < u_.size2(), "Index j out of range.");
    return u_(i,j);
}


std::vector<double> kron(const std::vector<double>& a, const std::vector<double>& b, double factor) {
    /*
    * 1D kronecker product.
        * Input: Vector, Vector, factor
        * Output: Vector
    */
    double m = a.size();
    double p = b.size();
    std::vector<double> out(m * p,0.0);

    for (size_t i = 0; i < m; ++i) {
            for (size_t k = 0; k < p; ++k) {
                    out[i*p+k] = a[i] * b[k] * factor;
                }
    }
    return out;
}

std::vector<Matrix> concatenate(const Matrix& A, const Matrix& C) {
    /*Trying to reproduce np.concatenate(A,C,2) from
        numpy library. 
    */
    std::vector<Matrix> B;//here maybe define the number of matrices already?? its only two...
    size_t rows = A.size1();
    size_t cols = A.size2();
    for (size_t c = 0; c < cols; ++c) {
        Matrix Binner(rows, 2.0);
        for (size_t r = 0; r < rows; ++r) {
            Binner(r, 0) = A(c, r);
            Binner(r, 1) = C(c, r);
        }
        B.push_back(Binner);
    }
    return B;
}


void alg_82(const int D, const std::vector<Matrix>& B, const std::vector<double>& G, std::vector<double>& v) {
    
    runtime_check(D != B.size(), "Dimension of problem and number of shape function matrices B building tensor space not equal!");

    size_t i_nq = B[0].size1();
    size_t i_nf = B[0].size2();

    alg_57(D, B, v);
    cie::spectralkernelhelpers::Vectorwise_Mult_helpers(G,v);

    std::vector<Matrix> B2 = swapaxes(B);

    alg_57(D, B2, v);
    //TODO: implement also algorithm 58
        //else:
        //    w1 = alg_58(D, B, v)
        //w2 = G * w1
        //r = alg_58(D, np.swapaxes(B, 0, 1), w2)
}

std::vector<double> alg_84(const int D, const std::vector<Matrix>& B, 
                            const std::vector<Matrix>& Bdiff, 
                            const std::vector<double>& M, 
                            const std::vector<double>& v){
    runtime_check(D != B.size(), "Dimension of problem and number of shape function matrices B or Bdiff building tensor space not equal!");

    size_t i_nq = B[0].size1();
    size_t i_nf = B[0].size2();

    std::vector<double> r(v.size(), 0.0);

    for (size_t d = 0; d < D; ++d) {
        std::vector<Matrix> Bd_diff = cie::spectralkernelhelpers::VectorMatrix_helpers(B.size(), i_nq, i_nf);
        std::vector<double> w1 = v;
        for (size_t i = 0; i < D; ++i) {
            if (i == d) {
                alg_84_mergeBd(Bd_diff, Bdiff,i);
            }
            else {
                alg_84_mergeBd(Bd_diff, B, i);
            }
        }

        alg_57(D, Bd_diff, w1);
        cie::spectralkernelhelpers::Vectorwise_Mult_helpers(M, w1);
        swapaxes(Bd_diff);
        // std::vector<Matrix> swapBd_diff = swapaxes(Bd_diff);
        alg_57(D, Bd_diff, w1);
        cie::spectralkernelhelpers::Vectorwise_Sum_helpers(r, w1);
    }
    return r;
}
void alg_84_mergeBd(std::vector<Matrix>& Bd_diff, const std::vector<Matrix>& Bdiff, const size_t t) {
    for (size_t i = 0; i < Bd_diff.size(); ++i) {
        for (size_t j = 0; j < Bd_diff[0].size1(); ++j) {
            Bd_diff[i](j, t) = Bdiff[i](j, t);
        }
    }
}

void alg_57(const int D, const std::vector<Matrix>& R, std::vector<double>& P) {
    /* here we want to create a copy, that is the reason behing not calling the vector P as
    a pointer.
    */

    runtime_check(D != R.size(), "Dimension of problem and number of matrices R  for Kronecker product not equal!");
    runtime_check(R[0].size1() != R[0].size2(), "Algorithm just works for quadratic matrices R!");

    int n_i = R[0].size1();
    int size = std::pow(n_i, D);
    std::vector<double> T(P.size(), 0.0);
    int stride = (size / n_i);
    // std::cout<<stride<<std::endl;

    for (int i = 0; i < D; ++i) {// in range(0, D) :
        for (int j = 0; j < stride; ++j) {//this creates 
            std::vector<double> Pfraction(P.begin() + j * n_i, P.begin() + (j + 1) * n_i);
            Matrix Rfraction(n_i, n_i);
            for (size_t Ri = 0; Ri < n_i; Ri++) {
                for (size_t Rj = 0; Rj < n_i; Rj++) {
                    Rfraction(Ri, Rj) = R[Ri](Rj, i);
                    // std::cout<<"Rfraction ["<<Ri<<"]["<<Rj<<"]"<<"R[][]["<<i<<"]"<<std::endl;
                }
            }
            int value_idx = 0;
            std::vector<double> val=Rfraction*Pfraction;//Matrix*vector
            for (int inner = j; inner < size; inner += stride) {
                T[inner] = val[value_idx++];
            }
            //# Fehler in Davide's Diss -> linke Seite beginnend bei j
        }
        std::swap(P, T);
    }
    // return P;
}

void swapaxes(std::vector<Matrix>& A) {
    // trying to reproduce np.swapaxes(A,0,1);
    // this is only if t and i is the same number!!!
    //     meaning this only works for algorithm57
    runtime_check(A.size() == A[0].size1(),
        "Dimension problem and number of Matrices not equal to number of rows!");

    std::vector<Matrix> Mout(A);//VectorMatrix(A.size(), A[0].size1(), A[0].size2());

    for (size_t t = 0; t < A.size(); ++t) {
        for (size_t i = 0; i < A[0].size1(); ++i) {
            for (size_t j = 0; j < A[0].size2(); ++j) {
                A[t](i, j) = Mout[i](t, j);
            }
        }
    }
}
std::vector<Matrix> swapaxes(const std::vector<Matrix>& A) {
    /* trying to reproduce np.swapaxes(A,0,1);
    this is only if t and i is the same number!!!
        meaning this only works for algorithm57
    */

    runtime_check(A.size() == A[0].size1(),
        "Dimension problem and number of Matrices not equal to number of rows!");

    std::vector<Matrix> Mout = VectorMatrix(A.size(), A[0].size1(), A[0].size2());

    for (size_t t = 0; t < A.size(); ++t) {
        for (size_t i = 0; i < A[0].size1(); ++i) {
            for (size_t j = 0; j < A[0].size2(); ++j) {
                Mout[t](i, j) = A[i](t, j);
            }
        }
    }
    return Mout;
}
std::vector<Matrix> VectorMatrix(int sv, int Msize1, int Msize2) {
    /*
    sv: # of matrices
    Function: creates a vector of matrices
    Msize1: # of rows in matrices
    Msize2: # of columns in matrices
    */
    std::vector<Matrix> Mout;
    for (size_t m = 0; m < sv; ++m) {
        Matrix innerM(Msize1, Msize2);
        Mout.push_back(innerM);
    }
    return Mout;
}

std::vector<double> Vectorwise_Mult(const std::vector<double>& R, std::vector<double>& s) {
    std::vector<double> Vout;
    for (size_t i = 0; i < R.size(); ++i) { Vout.push_back(R[i] * s[i]); }
    return Vout;
}
// void Vectorwise_Mult_light(const std::vector<double>& R, std::vector<double>& s) {
//     for (size_t i = 0; i < s.size(); ++i) {
//         s[i] = R[i] * s[i];
//     } 
// }
std::vector<double> Vectorwise_Sum(const std::vector<double>& R, std::vector<double>& s) {
    std::vector<double> Vout;
    for (size_t i = 0; i < R.size(); ++i) { Vout.push_back(R[i] + s[i]); }
    return Vout;
}
// void Vectorwise_Sum_light(const std::vector<double>& R, std::vector<double>& s) {
//     std::vector<double> Vout;
//     for (size_t i = 0; i < R.size(); ++i) { s[i] = R[i] + s[i]; }
// }
std::vector<double> ontheFly::slice(size_t it, const std::vector<double>& indices) const {
    // u_ holds every timestep only in the default mode; the storage modes
    // keep a single row. A row that does not exist must fail, not fall back to row 0.
    if (it >= u_.size1()) {
        throw std::out_of_range("ontheFly::slice: asked for row " + std::to_string(it)
            + " but u_ has " + std::to_string(u_.size1()) + " row(s)");
    }
    std::vector<double> u_slice;
    u_slice.reserve(indices.size());
    for (size_t index : indices) {
        u_slice.push_back(u_(it, static_cast<size_t>(index)));
    }
    return u_slice;
}

std::vector<double> slice(const std::vector<double>& u, const std::vector<double>& indices){
    std::vector<double> u_slice;
    for (size_t index : indices) {
        u_slice.push_back(u[index]);
    }
    return u_slice;
}

void writeU_inter(const Matrix& matrix, const std::string filename) {

    std::ofstream file(filename, std::ios::app); // Open in append mode
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    size_t size1 = matrix.size1(); // Number of rows
    size_t size2 = matrix.size2(); // Number of columns
    // std::ofstream outfile(filename);
    for (size_t i = 0; i < size1; ++i) {
        // Use the linalghelper::writeRow function to write each row
        cie::linalg::linalghelper::writeRow(
            [&](size_t j) { return matrix(i, j); }, size2, file, 12);
    }
    // No need for an extra newline, linalghelper::writeRow adds one.
}


void writeU_(Matrix& u, std::string name) {
    //this function might not be helpful anymore
    // std::cout << "Starting writing on file..." << std::endl;

    // std::ofstream outfile(name);
    // linalg::write(u, outfile);
    // outfile.close( );

    // std::cout << "Your file has been saved as: \n" << name << std::endl;
}



Matrix ontheFly::hardcodedwriteonu() {
    for (size_t it = 2; it <= nsteps_; ++it) {
        for (int dof = 0; dof < ndofglobal_; ++dof) {
            u_(it, dof) = it-1;
        }
    }
    return u_;
}

}//otFly

}//cie

