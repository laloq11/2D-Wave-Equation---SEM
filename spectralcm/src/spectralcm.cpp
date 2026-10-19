// #include "spectralcm.hpp" 

// #include <include/time.h>
#include "spectralcm.hpp" 
#include <cassert> 
#include <array>

namespace cie
{
namespace spectralkernel
{

    Params w_eq_2d_decreaseGrid() {
        // double eps = 1e-12;
        //initializing
        std::array<size_t, 2> nelements = { 5, 5 };
        std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
        std::array<double, 2>center = { 1.0, 1.0 };

        double nsteps = 1000;
        size_t polynomialDegree = 2;

        double duration = 12;
        double radius = 0.4;
        double frequency = 1.0;
        double sigmaS = 0.06;
        double E = 1.0;
        double rho = 1.0;
        double domain = 1.0;
        double alpha = 1.0;
        double c = 1.0;
        Params params(nelements, lengths, center, E, rho, domain, alpha, c,
            duration, radius, frequency, sigmaS,
            nsteps, polynomialDegree);
        // std::cout<<"number of DOFs:"<<params.ndofglobal()<<std::endl;
        return params;
    }
    Params w_eq_2d_completeGrid() {
        // double eps = 1e-12;
        //initializing
        std::array<size_t, 2> nelements = { 10, 10 };
        std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
        std::array<double, 2>center = { 1.0, 1.0 };

        double nsteps = 1000;
        size_t polynomialDegree = 2;

        double duration = 12;
        double radius = 0.4;
        double frequency = 1.0;
        double sigmaS = 0.06;
        double E = 1.0;
        double rho = 1.0;
        double domain = 1.0;
        double alpha = 1.0;
        double c = 1.0;
        Params params(nelements, lengths, center, E, rho, domain, alpha, c,
            duration, radius, frequency, sigmaS,
            nsteps, polynomialDegree);
        // std::cout<<"number of DOFs:"<<params.ndofglobal()<<std::endl;
        return params;
    }
    Params w_eq_2d_decreaseGrid_4testing(size_t nx, size_t ny, size_t PolDegree,
        double duration, size_t nsteps) {
        // double eps = 1e-12;
        //initializing
        std::array<size_t, 2> nelements = { nx, ny };
        std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
        std::array<double, 2>center = { 1.0, 1.0 };

        // size_t polynomialDegree = 2; //input
        // duration, nsteps: inputs (dt = duration / nsteps)
        double radius = 0.4;
        double frequency = 1.0;
        double sigmaS = 0.06;
        double E = 1.0;
        double rho = 1.0;
        double domain = 1.0;
        double alpha = 1.0;
        double c = 1.0;
        Params params(nelements, lengths, center, E, rho, domain, alpha, c,
            duration, radius, frequency, sigmaS,
            nsteps, PolDegree);
        // std::cout<<"number of DOFs:"<<params.ndofglobal()<<std::endl;
        return params;
    }
    Params w_eq_2d_autostable(size_t nx, size_t ny, size_t PolDegree, size_t nsteps) {
        // double eps = 1e-12;
        //initializing
        std::array<size_t, 2> nelements = { nx, ny };
        std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
        std::array<double, 2>center = { 1.0, 1.0 };
        
        // size_t polynomialDegree = 2; //input
        // duration, nsteps: inputs (dt = duration / nsteps)
        double radius = 0.4;
        double frequency = 1.0;
        double sigmaS = 0.06;
        double E = 1.0;
        double rho = 1.0;
        double domain = 1.0;
        double alpha = 1.0;
        double c = 1.0;
        
        // dt_crit * c / h, uncut square cell, GLL-lumped M, GLL-integrated K, p = 1..10
        static constexpr std::array<double, 10> Cp = {
            0.707106781186548, 0.288675134594813, 0.164054624686693, 0.104442278311696,
            0.071432315609361, 0.051655014512142, 0.039005419539448, 0.030464365739356,
            0.024438904279170, 0.020034146407630 };
        assert(nx == ny && PolDegree >= 1 && PolDegree <= Cp.size());

        double safety = 0.9;
        double h = lengths[0] / nx;
        double dt = safety * Cp[PolDegree - 1] * h / c;
        double duration = nsteps * dt;

        Params params(nelements, lengths, center, E, rho, domain, alpha, c,
            duration, radius, frequency, sigmaS,
            nsteps, PolDegree);
        // std::cout<<"number of DOFs:"<<params.ndofglobal()<<std::endl;
        return params;
    }
    // Matrix solve_w_eq_2d_decreaseGrid(bool w,std::string name) {
    //     double time1, timedif;        /* use doubles to show small values */
    //     time1 = (double)clock();            /* get initial time */
    //     time1 = time1 / CLOCKS_PER_SEC;      /*    in seconds    */

    //     std::cout << "\t--\tStarting simulation... at time " << time1 << std::endl;

    //     std::cout << "\t--\tInitializing parameters..." << std::endl;

    //     //initialize params from function
    //     Params params = cie::spectralkernel::w_eq_2d_decreaseGrid();

    //     std::cout << "\t--\tInitializing solver..." << std::endl;
    //     otFly::ontheFly solver(params.nsteps(), params.ndofglobal());

    //     std::cout << "\t--\tStarting solver..." << std::endl;
    //     Matrix solution1 = solver.otF_Solve(params, true);

    //     if (w) {
    //         std::cout << "\t--\tStarting log on file..." << std::endl;
    //         otFly::writeU_(solution1, name);
    //     }
    //     std::cout << "\t--\tExiting program..." << std::endl;

    //     timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
    //     std::cout << "\t--\tTotal time:\t" << timedif << std::endl;
    //     return solution1;
    // }

    void solve_w_eq_2d_decreaseGrid(bool w,std::string name, std::string s) {
        // double time1, timedif;        /* use doubles to show small values */
        // time1 = (double)clock();            /* get initial time */
        // time1 = time1 / CLOCKS_PER_SEC;      /*    in seconds    */

        // std::cout << "\t--\tStarting simulation... at time " << time1 << std::endl;

        std::cout << "\t--\tInitializing parameters..." << std::endl;

        //initialize params from function
        Params params = cie::spectralkernel::w_eq_2d_decreaseGrid();

        std::cout << "\t--\tInitializing solver..." << std::endl;
        
        std::cout << "\t--\tStarting solver..." << std::endl;
        if (s == "ontheFly"){
            otFly::ontheFly solver(params.nsteps(), params.ndofglobal());
            Matrix solution1 = solver.otF_Solve(params, true);
        }
        else if (s=="ontheFly_omp"){
            otFly::ontheFly solver(params.nsteps(), params.ndofglobal());

            Matrix solution1 = solver.otF_Solve_OMP(params, true);
        }
        else if (s=="ontheFly_omp_write"){
            std::cout<<
            "Starting ontheFly OMP with write function"<<std::endl;
            ontheFly solver(params.nsteps(), params.ndofglobal(),"storage");

            solver.otF_Solve_OMP_write(params,name,true, true);
            }
        else{
            Matrix solution1(1,0.0);
            std::cout<<"nothing done"<<std::endl;
        }

        std::cout << "\t--\tExiting program..." << std::endl;


    }

    Matrix testingPython_output() {

        //std::cout << params.ndofglobal() << std::endl;

        int nsteps = 100;
        int ndofglobal = 40;
        cie::otFly::ontheFly test(nsteps, ndofglobal);
        Matrix vector_Result = test.hardcodedwriteonu();

        return vector_Result;
    }

    



} // namespace splinekernel
// namespace ompkernel{
//     void solve_w_eq_2d_decreaseGrid(bool w,std::string name) {
//         // double time1, timedif;        /* use doubles to show small values */
//         // time1 = (double)clock();            /* get initial time */
//         // time1 = time1 / CLOCKS_PER_SEC;      /*    in seconds    */

//         // std::cout << "\t--\tStarting simulation... at time " << time1 << std::endl;

//         // std::cout << "\t--\tInitializing parameters..." << std::endl;

//         //initialize params from function
//         cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid();

//         std::cout << "\t--\tInitializing solver..." << std::endl;
//         otFly_omp::ontheFly solver(params.nsteps(), params.ndofglobal());

//         std::cout << "\t--\tStarting solver..." << std::endl;
//         Matrix solution1 = solver.otF_Solve(params, true);

//         if (w) {
//             std::cout << "\t--\tStarting log on file..." << std::endl;
//             otFly_omp::writeU_(solution1, name);
//         }
//         std::cout << "\t--\tExiting program..." << std::endl;

//         // timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
//         // std::cout << "\t--\tTotal time:\t" << timedif << std::endl;
//         // return solution1;
//     }
// } //namespace ompkernel

namespace performancekernel
{


    Performance::Performance() : t0_((double)clock()/CLOCKS_PER_SEC)
    {
        times_.push_back((((double)clock()) / CLOCKS_PER_SEC) - t0_);
        title_="";
    };
    
    Performance::Performance(std::string s) : t0_((double)clock()/CLOCKS_PER_SEC), title_(s)
    {
        times_.push_back((((double)clock()) / CLOCKS_PER_SEC) - t0_);
    };

    double Performance::time_Diff(size_t i) const { 
        if (i> times_.size()){
            std::cout<<"Error on the number requested"<<std::endl;
            return 0.0;
        }
        else{
            return times_[i];
        }
    }
    void Performance::checkpoint(){ 
        times_.push_back((((double)clock()) / CLOCKS_PER_SEC) - t0_);
         
        }
    Performance::~Performance( ){
        // std::cout<<"---------------Checkpoints start at:\t\t"  <<t0_<<"...\n\n";
        if (title_!=""){
            std::cout<<"---------------Checkpoints of "<<title_<<" are:"<<std::endl;
        }
        else{
            std::cout <<"Checkpoints are:\n\n";
        }
        for (size_t i=0;i<times_.size();i++){
            std::cout<<"Checkpoint #"<<i<<":\t\t"<<times_[i]<<"\t secs."<<std::endl;
        }
        std::cout<<"---------------Checkpoints of"<<title_<<" end...\n\n";
    }

}//performancekernel




} // namespace cie

