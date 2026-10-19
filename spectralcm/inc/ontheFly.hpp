#pragma once

#include <vector>
#include <tuple>
#include <array>
#include <functional>

#include "linalg.hpp"
#include "alias.hpp"
#include "params.hpp"

#include <omp.h>

namespace cie
{
    namespace otFly
    {
    using Params = cie::spectralkernel::Params;
    using Matrix= cie::linalg::Matrix;
    // using uniformMesh = spectralkernel::uniformMesh;

    class ontheFly {
    public:
        ontheFly(int nsteps, int ndofglobal);
        ontheFly(int nsteps, int ndofglobal, std::string mode);

        Matrix otF_Solve_2(const Params& params);
        Matrix otF_Solve(const Params& params, bool debug);
        Matrix hardcodedwriteonu();
        std::vector<double> otF_iteration_Solve(const Params& params, int it, std::vector<double>& u_1, std::vector<double>& u_2);

        int ndofglobal() const { return ndofglobal_; }
        void set_ndofglobal(const int ndfglbl) { ndofglobal_ = ndfglbl; }

        // std::vector<double> compute_invM_fKu(const Params& params, const double it);

        std::vector<double> operator()(size_t i) const;

        double operator()(size_t i, size_t j) const;

        void scatter(const std::vector<double>& vector2scatter, const LocationMap& locationMap, int dt);


        std::vector<double> compute_Mass_Cvector(const Params& params);

        std::vector<double> compute_fKu_Cvector(const Params& params,
            const size_t it);
        std::vector<double> compute_fKu_CvectorCAT(const Params& params,
                                                        const double it);
        // void compute_fKu_Cvector_OMPxCAT(std::vector<double>& fku,const Matrix& u,
        //     const Params& params, const double it);

        void compute_invM_fKuCAT(const Params& params, 
        const double it);

        Matrix otF_Solve_OMP(const Params& params, bool debug);
        void otF_Solve_OMP_write(const Params& params, std::string fname, 
            bool write, bool debug);
        // std::vector<double> compute_Mass_Cvector_OMP(const Params& params);
        // std::vector<double> compute_fKu_Cvector_OMP(const Params& params,
        //                                                 const double it) ;

        void compute_invM_fKu_OMP(const Params& params, const size_t it);
        // void compute_fKu_Cvector_OMP(const Params& params, const double it);

        void compute_invM_fKu(const Params& params, const size_t it);

        void fKu_initialize(std::vector<double>& fKu);

        std::vector<double> slice(size_t it, const std::vector<double>& indices) const;
        // void compute_Mass_Cvector_OMP(const Params& params);

        // double* fku() const {return fku_.data();} 
        Matrix get_u() const { return u_; };
        // std::vector<double> getlast_Uv();
        std::vector<double> get_up(int it) {return up1_;}

        // Matrix& get_u() const { return u_; };
        std::vector<double> get_fku() const {return fku_;}
        // std::vector<double> get_fku() {return fku_;}
        std::vector<double>& get_fku() {return fku_;}
        // std::vector<double>& fku()  {return fku_;}
        std::vector<double> get_mu() const {return Mu_;}
        std::vector<double>& get_mu() {return Mu_;}
        Matrix otF_SolveCat(const Params& params, bool debug);
        Matrix otF_SolveOMPxCat(const Params& params, bool debug);

        void updated_us();

    private:
        int nsteps_, ndofglobal_;
        Matrix u_;
        std::vector<double> fKu_invM_;
        std::vector<double> fku_,Mu_;
        std::vector<double> up1_,up2_;
        };

    std::vector<double> kron(const std::vector<double>& a, const std::vector<double>& b, double factor);
    std::vector<Matrix> concatenate(const Matrix& A,const Matrix& C);
    // std::vector<double> alg_57(const int D, std::vector<Matrix>& R, std::vector<double> P);
    
    // std::vector<Matrix> swapaxes(const std::vector<Matrix>& A);
    void swapaxes(std::vector<Matrix>& A);
    std::vector<Matrix> swapaxes(const std::vector<Matrix>& A);
    
    std::vector<double> Vectorwise_Mult(const std::vector<double>& R, std::vector<double>& s);
    std::vector<double> Vectorwise_Sum(const std::vector<double>& R, std::vector<double>& s);
    // std::vector<double> alg_82(const int D, std::vector<Matrix>& B, const std::vector<double>& G, std::vector<double> v);
    // std::vector<double> alg_84(const int D, std::vector<Matrix>& B, 
    //                             const std::vector<Matrix>& Bdiff, 
    //                             const std::vector<double>& M, 
    //                             std::vector<double> v);
    std::vector<Matrix> VectorMatrix(int sv, int sM1, int sM2);
    void alg_84_mergeBd(std::vector<Matrix>& Bd_diff, const std::vector<Matrix>& Bdiff, const size_t t);

    void writeU_(Matrix& u, std::string name);
    std::vector<double> alg_84(const int D, 
                        const std::vector<Matrix>& B, 
                        const std::vector<Matrix>& Bdiff, 
                        const std::vector<double>& M, 
                        const std::vector<double>& v);

    void alg_57(const int D, const std::vector<Matrix>& R,
         std::vector<double>& P);
    void alg_82(const int D, const std::vector<Matrix>& B,
         const std::vector<double>& G, std::vector<double>& v);
    
    //omp
    void compute_fKu_Cvector_OMP(std::vector<double>& fku,
        const std::vector<double>& u_p1,const Params& params,
         const double it);    
    void compute_fKu_Cvector_OMPxCAT(std::vector<double>& fku,
        const Matrix& u,
        const Params& params, const double it);
    void compute_Mass_Cvector_OMP(std::vector<double>& mu,
        const Params& params);


    std::vector<double> slice(const std::vector<double>& u, 
        const std::vector<double>& indices);
    void writeU_inter(const Matrix& matrix, 
        const std::string filename);

    }//spectralkernel
}//cie