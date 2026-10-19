#include <params.hpp>

#include <stdexcept>

namespace cie
{
namespace spectralkernel
{

Params::Params(std::array<size_t, 2> nelements,
                std::array<double, 2> lengths, 
                std::array<double, 2> center, 
                double E, double rho, double domain, double alpha, double c,
                double duration,
                 double radius, 
                double frequency, double sigmaS,
                int nsteps, size_t polynomialDegree) 
        {
        domain_ = domain;
        lengths_ = lengths;
        nelements_ = nelements;
        duration_ = duration;

        alpha_ = alpha;
        E_ = E;
        rho_ = rho;

        eps_ = 1e-10;

        plotfreq_ = 50; //change this to display on plot frequency
        center_ = center;
        radius_ = radius;
        dt_ = duration / nsteps;
        nsteps_ = nsteps;
        dx_ = lengths[0] / nelements[0];
        dy_ = lengths[1] / nelements[1];
        quadratureOrder_ = polynomialDegree + 1;
        polynomialDegree_ = polynomialDegree;

        adhocpp::numeric::GaussLobatto gll(quadratureOrder_);
        
        gllCoordinates_ = gll.getCoordinates();
        gllWeights_ = gll.getWeights();

        ndofelement_ = std::pow(polynomialDegree + 1,2);
        ndofdirection_ = { nelements[0] * polynomialDegree + 1, nelements[1] * polynomialDegree + 1 };
        ndofglobal_ = ndofdirection_[0] * ndofdirection_[1];
        c_ = c;
        double pi = 3.14159265358979323846;
        t0_ = 1.0 / frequency;
        sigmaT_ = 1.0 / (2.0 * pi * frequency);
        sigmaS_ = sigmaS;
        // Excitement
        //lambda functions
        mapX_ = lambdamapXY(dx_); //mapX(ielement,r)
        mapY_ = lambdamapXY(dy_); //mapX(ielement,r)
        fx_ = lambdafx(sigmaS_);  //fx(x,y) coordinates
        ft_ = lambdaft(sigmaT_, t0_); //ft(t) time

        shapesDiff0_ = cie::spectralkernel::get_ShapesDiff(gllCoordinates_, 0);
        shapesSDiff1_ = cie::spectralkernel::get_ShapesDiff(gllCoordinates_, 1)* (2.0 / dx_);
        shapesRDiff1_ =cie::spectralkernel::get_ShapesDiff(gllCoordinates_, 1)* (2.0 / dy_);

        B_ = cie::otFly::concatenate(shapesDiff0_, shapesDiff0_);
        Bdiff_ = cie::otFly::concatenate(shapesSDiff1_,shapesRDiff1_);
        M_ = cie::otFly::kron(gllWeights_, gllWeights_,
                (rho_ * (dx_ / 2.0) * (dy_ / 2.0) * c_ * c_ ));
        G_ = cie::otFly::kron(gllWeights_, gllWeights_,
                ( rho_ * ( dx_/ 2.0 ) * ( dy_ / 2.0) ) );

/*
        MISSING LAMDA FUNCTIONS:
            //treedepth = polynomialDegree + 1
            //nseedpoints = polynomialDegree + 2
            //equidistantCoordinates = np.linspace(-1.0, 1.0, polynomialDegree + 1)//do we need this?
*/

} // constructor

Matrix Params::solve(std::string solver_type, std::string filename) {
    //this function is mainly for python binding.
    Matrix u_out;
    if (solver_type == "ontheFly") {

        std::cout << "\t--\tInitializing solver..." << std::endl;
        ontheFly solver((*this).nsteps(), (*this).ndofglobal());

        std::cout << "\t--\tStarting solver..." << std::endl;
        u_out = solver.otF_Solve((*this),true);
        std::cout << "\t--Returning values to python script..." << std::endl;
    }
    else if (solver_type=="ontheFly_omp"){
        std::cout << "\t--\tInitializing solver..." << std::endl;
        ontheFly solver((*this).nsteps(), (*this).ndofglobal());

        std::cout << "\t--\tStarting solver..." << std::endl;
        u_out = solver.otF_Solve_OMP((*this),true);
        std::cout << "\t--Returning values to python script..." << std::endl;
    }

    
    else if (solver_type=="ontheFly_omp_write"){
        std::string mode = "storage";
        std::cout << "\t--\tInitializing solver..." << std::endl;
        ontheFly solver((*this).nsteps(), (*this).ndofglobal(),mode);
        
        std::cout << "\t--\tStarting solver..." << std::endl;
        solver.otF_Solve_OMP_write((*this),filename,true,true);

        //not returning anything. Writing output on file...
        Matrix u_null(1,0.0);
        //returning the least possible data because writing was performed
        return u_null;
    }
    else if (solver_type=="cu_singleThread"){
        #ifdef CUDA_ON
        cie::otFlyGPU::check_polynomialDegree_gpu((*this).polynomialDegree());

        std::cout << "\t--\tInitializing CUDA solver a SINGLE thread per element..." << std::endl;
        // ontheFly solver((*this).nsteps(), (*this).ndofglobal());
        cie::otFlyGPU::solve_w_eq_2d_cu_singleThread<float>(*this,true,filename);
        std::cout << "\t--\tStarting solver..." << std::endl;
        Matrix u_null(1,0.0);
        return u_null;
        #else
        throw std::runtime_error("Solver 'cu_singleThread' requires CUDA. Rebuild with CUDAToolkit available.");
        #endif

    }
    else if (solver_type=="cu_pDegreeThread"){
        #ifdef CUDA_ON
        cie::otFlyGPU::check_polynomialDegree_gpu((*this).polynomialDegree());

        std::cout << "\t--\tInitializing CUDA solver using (pD+1)^2 threads per element..." << std::endl;
        // ontheFly solver((*this).nsteps(), (*this).ndofglobal());
        cie::otFlyGPU::solve_w_eq_2d_cu_pDegreeThread<float>(*this,true,filename);
        std::cout << "\t--\tStarting solver..." << std::endl;
        Matrix u_null(1,0.0);
        return u_null;
        #else
        throw std::runtime_error("Solver 'cu_pDegreeThread' requires CUDA. Rebuild with CUDAToolkit available.");
        #endif

    }
    else {
        throw std::invalid_argument("Params::solve: unknown solver \"" + solver_type
            + "\"; valid: ontheFly, ontheFly_omp, ontheFly_omp_write, cu_singleThread, cu_pDegreeThread");
    }
    return u_out;
}

// template <typename ValueType>
std::vector<double> Params::createLocationMap(int ielement, int jelement) const { 
    std::vector<double> dofIndicesI = arange(polynomialDegree_ + 1, polynomialDegree_ * ielement);
    std::vector<double> dofIndicesJ = arange(polynomialDegree_ + 1, polynomialDegree_ * jelement);
    std::vector<double> LocationMap;

    for (size_t i = 0; i <= polynomialDegree_; ++i) {
        for (size_t j = 0; j <= polynomialDegree_; ++j) {
            LocationMap.push_back(dofIndicesI[i]*ndofdirection_[1] + dofIndicesJ[j]);
        }
    }
    return LocationMap;
}
// template std::vector<double> Params::createLocationMap(int, int) const ;
// template std::vector<size_t> Params::createLocationMap(int, int) const ;


std::vector<double> Params::get_Force_xy(int ielement, int jelement) const {

    std::vector<double> u_m(ndofelement_, 0.0);
    // Evaluate Lagrange polynomials in both directions
    for (size_t igauss = 0; igauss < gllCoordinates_.size(); ++igauss) {
        double r = gllCoordinates_[igauss];
        for (size_t jgauss = 0; jgauss < gllCoordinates_.size(); ++jgauss) {
            double s = gllCoordinates_[jgauss];
            double ii = igauss * gllCoordinates_.size() + jgauss;
            double x = mapX_(ielement, r); 
            double y = mapY_(jelement, s);
            // Add contributions
            u_m[ii] = fx_(x, y);
    }
}
    return u_m;
}

// template <typename ValueType>
// void Params::get_global_Forces(std::vector<ValueType>& globalF) const {
//     // size_t ndofelemlocal = gllCoordinates_.size()*gllCoordinates_.size();
//     // std::cout<<"This two must be equal"<<
//     // ndofelemlocal<<", "<<ndofelement_<<std::endl;
    
//     // size_t total_elements = nelements_[0]*nelements_[1];

//     // std::vector<ValueType> globalF(ndofglobal_, 0.0);

//     for(int iel=0;iel<nelements_[0];iel++){
//         for(int jel=0;jel<nelements_[1];jel++){

//             std::vector<double> locationMap = createLocationMap(iel,jel);

//             for (size_t igauss = 0; igauss < gllCoordinates_.size(); ++igauss) {
//                 double r = gllCoordinates_[igauss];
//                 for (size_t jgauss = 0; jgauss < gllCoordinates_.size(); ++jgauss) {
//                     double s = gllCoordinates_[jgauss];
//                     double ii = igauss * gllCoordinates_.size() + jgauss;
//                     double x = mapX_(iel, r); 
//                     double y = mapY_(jel, s);
//                     ValueType gF = fx_(x, y);
//                     // Add contributions
//                     // std::cout<<"global Forces element("<<iel<<","<<jel<<"), lM["<<locationMap[ii]<<"], sub ("<<igauss<<","
//                     // <<jgauss<<")= ";
//                     globalF[locationMap[ii]] = gF;
//                     // std::cout<<globalF[locationMap[ii]]<<std::endl;
//                 }
//             }
//         }
//     }
//     // return globalF;
// }
// template void Params::get_global_Forces(std::vector<double>&) const;
// template void Params::get_global_Forces(std::vector<float>&) const;


SpatialFunction Params::lambdamapXY(const double ds) const {
    return [=](int selement, double s) {
        return (selement + s / 2.0 + 0.5) * ds;;
        };
}

SpatialFunction Params::lambdafx(const double sigmaS) const {
    return [=](double x,double y) {
        return 10.0 * std::exp( 
            -1.0 * ( std::pow(x, 2) + std::pow(y, 2) )  / ( 2.0 * std::pow(sigmaS, 2)) 
        );;
        };
}

TimeFunction Params::lambdaft(const double sigmaT, const double t0) const {
    return [=](double t) {
        const double pi = 3.14159265358979323846;
        return -(t - t0) / (std::sqrt(2.0 * pi) * std::pow(sigmaT, 3)) *
            std::exp( -1.0 * std::pow( (t - t0) , 2 ) / ( 2.0 * std::pow( sigmaT, 2 ) ) );;
        };
}

Matrix get_ShapesDiff(const std::vector<double>& coordinates,const size_t diff){
    adhocpp::numeric::LagrangeBasisFunctions shapes0(coordinates);
    std::vector<std::vector<double>> shapesDiff0;
    // Matrix shapesDiff0(coordinates.size(),coordinates.size(),0.0);

    for (size_t coor = 0; coor < coordinates.size(); ++coor) {
        std::vector<double> shapesDiff0_r;
        for (size_t shape = 0; shape < coordinates.size(); ++shape) {
            if (shape != 1) {
                shapesDiff0_r.push_back(shapes0.evaluateSingle(shape, 0, diff, coordinates[coor]));
            }
        }
        shapesDiff0_r.push_back(shapes0.evaluateSingle(1, 0, diff, coordinates[coor]));
        shapesDiff0.push_back(shapesDiff0_r);
    }
    return shapesDiff0;
}

Matrix get_ShapesDiffnew(const std::vector<double>& coordinates,const size_t diff){
    adhocpp::numeric::LagrangeBasisFunctions shapes0(coordinates);
    // std::vector<std::vector<double>> shapesDiff0;
    Matrix shapesDiff0(coordinates.size(),coordinates.size(),0.0);

    for (size_t coor = 0; coor < coordinates.size(); ++coor) {
        std::vector<double> shapesDiff0_r;
        for (size_t shape = 0; shape < coordinates.size(); ++shape) {
            if (shape == 1) {
                // shapesDiff0_r.push_back(shapes0.evaluateSingle(shape, 0, diff, coordinates[coor]));
                shapesDiff0(coor,shape-1) = shapes0.evaluateSingle(shape, 0, diff, coordinates[coor]);
                shapesDiff0(coor,coordinates.size()-1) = shapes0.evaluateSingle(shape, 0, diff, coordinates[coor]);
            }
            else if (shape <1){
                shapesDiff0(coor,shape) = shapes0.evaluateSingle(shape, 0, diff, coordinates[coor]);
            }
            else {
                shapesDiff0(coor,shape-1) = shapes0.evaluateSingle(shape, 0, diff, coordinates[coor]);
            }
        }
    }
    return shapesDiff0;
}

std::vector<double> Params::getUniformMeshdata() const{
    std::vector<double> Vout(6,0.0);
    Vout[0] = ndofdirection_[0];
    Vout[1] = ndofdirection_[1];
    Vout[2] = center_[0]-lengths_[0]/2;
    Vout[3] = center_[1]-lengths_[1]/2;
    Vout[4] = lengths_[0]/ndofdirection_[0];
    Vout[5] = lengths_[1]/ndofdirection_[1];
    return Vout;
}
template <typename ValueType>
std::vector<ValueType> Params::Bvector() const { 
    std::vector<ValueType> out;
    for (size_t i =0; i<B_.size();i++){
        for (size_t j =0; j<B_[0].size1();j++){
            for (size_t k =0; k<B_[0].size2();k++){
                out.push_back(B_[i](j,k));
            }
        }
    }
    return out;
}
template std::vector<float> Params::Bvector<float>() const;
template std::vector<double> Params::Bvector<double>() const;

template <typename ValueType>
std::vector<ValueType> Params::Bdiffvector() const { 
    std::vector<ValueType> out;
    for (size_t i =0; i<Bdiff_.size();i++){
        for (size_t j =0; j<Bdiff_[0].size1();j++){
            for (size_t k =0; k<Bdiff_[0].size2();k++){
                out.push_back(Bdiff_[i](j,k));
            }
        }
    }
    return out;
}
template std::vector<float> Params::Bdiffvector<float>() const;
template std::vector<double> Params::Bdiffvector<double>() const;


} // namespace splinekernel
} // namespace cie