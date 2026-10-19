#pragma once

#include <vector>
#include <tuple>
#include <array>
#include <functional>
#include <iostream>

#include "linalg.hpp"
#include "alias.hpp"
#include "LagrangeBasisFunctions.hpp" 
#include "GaussLobatto.hpp" 
#include "basisfunctions.hpp" 
#include "ontheFly.hpp" 

#ifdef CUDA_ON
#include "cuontheFly.cuh"
#endif

namespace cie
{
    namespace spectralkernel
    {
        class Params
        {
        public:
            Params(std::array<size_t, 2> nelements,
                std::array<double, 2> lengths,
                std::array<double, 2> center,
                double E, double rho, double domain, double alpha, double c,
                double duration,
                double radius,
                double frequency, double sigmaS,
                int nsteps, size_t polynomialDegree);

            Matrix solve(std::string solver, std::string filename="");


            std::vector<double> createLocationMap(int ielement, int jelement) const;

            std::vector<double> get_Force_xy(int ielement, int jelement) const;

            double ft(double t) const { return ft_(t); }
            double fx(double x, double y) const { return fx_(x, y); }

            double gllcoords(const int i) const { return gllCoordinates_[i]; }
            double gllWeights(const int i) const { return gllWeights_[i]; }

            std::vector<double> gllcoords() const { return gllCoordinates_; }
            std::vector<double> gllcoords2py() const { return gllCoordinates_; }
            std::vector<double> gllWeights() const { return gllWeights_; }
            std::array<double, 2> lengths() const { return lengths_; }
            std::array<double, 2> center() const { return center_; }

            size_t nelements(size_t dir) const { return nelements_[dir]; }
            int ndofglobal() const { return ndofglobal_; }
            size_t ndofelement() const { return ndofelement_; }
            int nsteps() const { return nsteps_; }
            double dx() const { return dx_; }
            double dy() const { return dy_; }
            double mapX(int ielement, double r) const { return mapX_(ielement, r); }
            double mapY(int jelement, double s) const { return mapY_(jelement, s); }

            double dt() const { return dt_; }
            double rho() const { return rho_; }
            double c() const { return c_; }
            int samplingFrecuency() const { return plotfreq_; }
            void changesamplingFrequency(int amount) {
                plotfreq_=amount;
                }
            std::array<size_t, 2> ndofdirection() const { return ndofdirection_; }
            size_t polynomialDegree() const { return polynomialDegree_; }

            double radius() const { return radius_; }

            Matrix shapesDiff0() const { return shapesDiff0_; }
            Matrix shapesSDiff1() const { return shapesSDiff1_; }
            Matrix shapesRDiff1() const { return shapesRDiff1_; }
            std::vector<Matrix> B() const { return B_; }

            template <typename ValueType>
            std::vector<ValueType> Bvector() const;


            template <typename ValueType>
            std::vector<ValueType> Bdiffvector() const;

            std::vector<Matrix> Bdiff() const { return Bdiff_; }
            std::vector<double> M() const {return M_; }
            std::vector<double> G() const {return G_; }
            std::vector<double> getUniformMeshdata()const;
            double t0()const{  return t0_;}
            double sigmaT() const{  return sigmaT_;}
            double sigmaS() const{  return sigmaS_;}

            // template <typename ValueType>
            // void get_global_Forces(std::vector<ValueType>& globalF) const ;

        private:
            std::array<size_t, 2> nelements_, ndofdirection_;
            std::array<double, 2> lengths_, center_;
            std::vector<double> gllCoordinates_, gllWeights_;
            double duration_,eps_,dt_,dx_,dy_ ;
            double radius_, rho_,domain_,alpha_, c_, E_;
            int ndofglobal_,nsteps_,plotfreq_,quadratureOrder_;
            size_t polynomialDegree_, ndofelement_;
            Matrix shapesDiff0_; //= cie::spectralkernel::get_ShapesDiff(coordinates, 0);
            Matrix shapesSDiff1_,shapesRDiff1_; //= cie::spectralkernel::get_ShapesDiff(coordinates, 1);//* 2 / dx
            SpatialFunction mapX_, mapY_, fx_;
            double sigmaT_, t0_, sigmaS_;

            SpatialFunction lambdamapXY(const double dx) const;
            SpatialFunction lambdafx(const double sigmaS) const;
            TimeFunction ft_;
            TimeFunction lambdaft(const double sigmaT, const double t0) const;
            std::vector<Matrix> B_,Bdiff_;
            std::vector<double> M_,G_;
            /*
            List of necessary lambda functions

            mandatory:
                lagrange
                    shapesDiff0
                    shapesDiff1
                fx
                ft

            so far being equal to 1:
                domain
                alpha
                rho
                E

            */

        };
        Matrix get_ShapesDiff(const std::vector<double>& coordinates, const size_t diff);
        Matrix get_ShapesDiffnew(const std::vector<double>& coordinates,const size_t diff);
    } // namespace splinekernel
} // namespace cie

namespace cie
{
namespace spectralkernelhelpers
{
using Matrix= cie::linalg::Matrix;

inline void Vectorwise_Mult_helpers(const std::vector<double>& R, std::vector<double>& s) {
    for (size_t i = 0; i < R.size(); ++i) { s[i] *= R[i]; }
}
inline void Vectorwise_Sum_helpers(std::vector<double>& r, const std::vector<double>& s) {
    for (size_t i = 0; i < r.size(); ++i) { (r[i] += s[i]); }
}
inline std::vector<Matrix> VectorMatrix_helpers(size_t sv, int Msize1, int Msize2) {
    /*
    sv: # of matrices
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
}
}