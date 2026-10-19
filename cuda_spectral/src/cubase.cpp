
#include "cubase.hpp"
#include "cuontheFly.cuh"

#include <fstream>   // For file handling
#include <algorithm>


namespace cie {
namespace otFlyGPU {

namespace reference {
template <typename ValueType>
void compute_fku_ref(const size_t it,const cie::spectralkernel::Params& params,
    const std::vector<ValueType>& u_tm1, std::vector<ValueType>& fku) {

    std::fill(fku.begin(),fku.end(),0.0);

    ValueType ft_dt = params.ft(it * params.dt());

	ValueType rho = params.rho();//1.0;
	ValueType dx =  params.dx();
	ValueType dy =  params.dy();
    ValueType c = params.c();
	ValueType facG = (rho * (dx/2.0) * (dy/2.0));
	ValueType facM = (rho * (dx/2.0) * (dy/2.0))* c * c;
	ValueType facS = (2.0/dx);
	ValueType facR = (2.0/dy);
    // B
    
    //memory alloc
    size_t polynomialDegree = params.polynomialDegree();
    //dont allocate
    adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1); 

        // end dont allocate
    size_t ndofelement = params.ndofelement();
    size_t nelements = params.nelements(0)*params.nelements(1);
    size_t nelements0 = params.nelements(0);
    size_t ndofdirection1 = params.ndofdirection()[1];
    
	ValueType* M = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(M,gauss.getWeights(), gauss.getWeights(), facM , (polynomialDegree+1));

	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);


	Matrixt<ValueType> Sdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	Matrixt<ValueType> Rdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	cie::otFlyGPU::ShapeFunctionFactor(Sdiff,facR);
	cie::otFlyGPU::ShapeFunctionFactor(Rdiff,facS);


	ValueType* B = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree+1,2); //pD+1,dim 
	ValueType* Bdiff = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(Bdiff, Sdiff, Rdiff,polynomialDegree+1,2); //pD+1,dim 

	ValueType* G = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(G,gauss.getWeights(), gauss.getWeights(), facG , (polynomialDegree+1));

    size_t vsize = (polynomialDegree + 1)*(polynomialDegree + 1);

    std::vector<ValueType> global_Forces(params.ndofglobal(), 0.0); 
    get_global_ForcesV(params,global_Forces);

    // get_global_Forces();//params.get_Force_xy(ielement, jelement);

    // #pragma omp parallel for collapse (2)
    ValueType* Fe_element = new ValueType[vsize];
    ValueType* Fi_element = new ValueType[vsize];
    ValueType* u_m = new ValueType[vsize];

    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {

            std::vector<size_t> locationMap = createLocationMapcu(polynomialDegree,
                                                                ndofdirection1,
                                                                ielement, 
                                                                jelement);

            for (size_t ri=0;ri<vsize;ri++) Fi_element[ri]=0;
            slicecu<ValueType>(u_m, u_tm1.data(), locationMap,vsize);

            alg_84_flat<ValueType>(2, B,Bdiff, M, u_m,polynomialDegree,Fi_element);
                // std::cout<<"global Forces element("<<ielement<<","<<jelement<<
                // "), ";

            get_elementforce<ValueType>(Fe_element,global_Forces.data(),
            locationMap.data(),ndofelement);

            std::vector<double> Fe_elementgood = params.get_Force_xy(ielement, jelement);
            // std::cout<<"Fe good , testing= [\n";
            // for(size_t i =0;i<ndofelement;i++) std::cout<<Fe_elementgood[i]<<",and "<<Fe_element[i]<<std::endl;
            // std::cout<<"]"<<std::endl;

            alg_82_flat<ValueType>(2, B, G, Fe_element,polynomialDegree);
            // if (ielement==0 && jelement==0){
            //     std::cout<<"normalFe"<<std::endl;
            //     for (size_t si =0;si<ndofelement;si++) std::cout<<","<<Fe_element[si];
            //     std::cout<<"]"<<std::endl;
            //     std::cout<<"normalFi"<<std::endl;
            //     for (size_t si =0;si<ndofelement;si++) std::cout<<","<<Fi_element[si];
            //     std::cout<<"]"<<std::endl;
            //     std::cout<<"normal dt^2: "<<params.dt()<<", ft: "<<ft_dt<<std::endl;
            // }

            for (size_t i = 0; i < ndofelement; ++i) {
                // #pragma omp atomic

                ValueType val = std::pow(params.dt(), 2) * (Fe_element[i] * ft_dt - Fi_element[i]);
                // if (ielement==0 && jelement==0) std::cout<<val<<",";
                fku[locationMap[i]] += val; 
            }

            // if (ielement==0 && jelement==0){
            //     std::cout<<"\nAFTER\n normalfku "<<std::endl;
            //     for (size_t m = 0;m<ndofelement;m++) std::cout<<fku[locationMap[m]]<<",";; 
            //     std::cout<<", ";
            //     std::cout<<"]\n-----------------------------"<<std::endl;
            // }
        }
    }
    delete[] Fe_element;
    delete[] Fi_element;
    delete[] u_m;
}



template void compute_fku_ref<float>(const size_t it,  const cie::spectralkernel::Params&,
        const std::vector<float>&,std::vector<float>&);
template void compute_fku_ref<double>(const size_t it,  const Params&,
        const std::vector<double>&,std::vector<double>&);

template <typename ValueType>
void get_elementforce(ValueType* Fe,const ValueType* globalF,
     const size_t* locationM, const size_t ndofelement){
    for (size_t si =0;si<ndofelement;si++) Fe[si] = globalF[locationM[si]];
}
// template void get_elementforce<double>(const double*,double*,const size_t,const size_t,const size_t,const size_t);
template void get_elementforce<float>(float*,const float*,const size_t*,const size_t);
} // namespace reference


template <typename ValueType>
void get_global_ForcesV(const Params& params,std::vector<ValueType>& globalF) {

    for(int iel=0;iel<params.nelements(0);iel++){
        for(int jel=0;jel<params.nelements(1);jel++){

            std::vector<double> locationMap = params.createLocationMap(iel,jel);

            for (size_t igauss = 0; igauss < params.polynomialDegree()+1; ++igauss) {
                double r = params.gllcoords()[igauss];
                for (size_t jgauss = 0; jgauss < params.polynomialDegree()+1; ++jgauss) {
                    double s = params.gllcoords()[jgauss];
                    double ii = igauss * (params.polynomialDegree()+1) + jgauss;
                    double x = params.mapX(iel, r); 
                    double y = params.mapY(jel, s);
                    ValueType gF = params.fx(x, y);
                    // Add contributions
                    // std::cout<<"global Forces element("<<iel<<","<<jel<<
                    // "), lM["<<locationMap[ii]<<"], sub ("<<igauss<<","
                    // <<jgauss<<")= ";
                    globalF[locationMap[ii]] = gF;
                }
            }
        }
    }
}
template void get_global_ForcesV(const cie::spectralkernel::Params&,std::vector<double>&);
template void get_global_ForcesV(const cie::spectralkernel::Params&,std::vector<float>&);



template <typename ValueType>
void slicecu(ValueType* u_slice,const ValueType* u, const std::vector<size_t>& indices, size_t ndofelement){
    for (size_t si =0;si<ndofelement;si++) {
        u_slice[si] = u[indices[si]];
    }
}
template void slicecu<double>(double*,const double*, const std::vector<size_t>&, size_t);
template void slicecu<float>(float*, const float*, const std::vector<size_t>&, size_t);



namespace reference {
template <typename ValueType>
void compute_Mass_ref(std::vector<ValueType>& mu,
    const Params& params) {

    std::fill(mu.begin(),mu.end(),0.0);
    
    ValueType fac =  params.rho() * ( params.dx()/ 2.0 ) * ( params.dy() / 2.0);
    ValueType dif = 0.0;
    // B
    
    //memory alloc
    size_t polynomialDegree = params.polynomialDegree();
    //dont allocate
    adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1); 
    Matrixt<ValueType> S = get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);
    Matrixt<ValueType> R = get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);
        // end dont allocate
    size_t ndofelement = params.ndofelement();
    size_t ndofdirection1 = params.ndofdirection()[1];
    std::vector<Matrixt<ValueType>> B = concatenate_matrices(S, R);
    
    std::vector<ValueType> G = kroncu(gauss.getWeights(), gauss.getWeights(),fac);
    //allocate for std::vector<ValueType> M_el(ndofelement,1.0);

    // #pragma omp parallel for collapse (2)
    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {

            std::vector<ValueType> M_el(ndofelement,1.0);
        
            alg_82_matrices<ValueType>(2, B, G, M_el);

            std::vector<size_t> locationMap = createLocationMapcu(polynomialDegree,
                                                                ndofdirection1,
                                                                ielement, 
                                                                jelement);
            for (size_t i = 0; i < locationMap.size(); ++i) {
                // #pragma omp atomic
                mu[locationMap[i]] += M_el[i];
            }
        }
    }
}
template void compute_Mass_ref<float>(std::vector<float>&, const cie::spectralkernel::Params&);
template void compute_Mass_ref<double>(std::vector<double>&, const cie::spectralkernel::Params&);

template <typename ValueType>
void alg_84_flat(const size_t D, const ValueType* B, 
    ValueType* Bdiff, 
    const ValueType* M, 
    const ValueType* v,size_t pD, ValueType* r){
    // runtime_check(D != B.size(), "Dimension of problem and number of shape function matrices B or Bdiff building tensor space not equal!");

    // size_t i_nq = B[0].size1();
    // size_t i_nf = B[0].size2();
    size_t vsize=(pD+1)*(pD+1);
    // ValueType* r = new ValueType[vsize];
    // for (size_t ri=0;ri<vsize;ri++) r[ri]=0;
    // std::cout<<"in alg_84"<<std::endl;

    ValueType* w1 = new ValueType[vsize];
    ValueType* Bd_diff= new ValueType[vsize*D];
    // std::cout<<"normal alg 84"<<std::endl;
    for (size_t di = 0; di < D; ++di) {
        for(size_t ri=0;ri<vsize;ri++) w1[ri]=v[ri];

        for(size_t i = 0; i < D; ++i) {
            if (i == di) {
                alg_84_mergeBd_flat(Bd_diff, Bdiff,i,pD+1,pD+1,D);
                // std::cout<<di<<"<-di, i = "<<i<<std::endl;

            }
            else {
                alg_84_mergeBd_flat(Bd_diff, B, i,pD+1,pD+1,D);
                // std::cout<<"i = "<<i<<std::endl;

            }
        }

        // alg_57(D, Bd_diff, w1);
        alg_57_flat<ValueType>(D, Bd_diff, w1,pD,vsize);
        // std::cout<<"57.1 w1=[";
        // for(size_t ri=0;ri<vsize;ri++) std::cout<<w1[ri]<<",";
        // std::cout<<"]"<<std::endl;

        // cie::spectralkernelhelpers::Vectorwise_Mult_helpers(M, w1);
        Vectorwise_Mult_helpers_flat(M, w1,vsize);
        // std::cout<<"vectorwisemult w1=[";
        // for(size_t ri=0;ri<vsize;ri++) std::cout<<w1[ri]<<",";
        // std::cout<<"]"<<std::endl;

        // swapaxes(Bd_diff);
        ValueType* Bd_diff2 = swapaxes_flat(Bd_diff,D,pD+1,pD+1);
        //for now we are using a copy. But most likely to change this 
        // to use Bd_diff itself.
        // alg_57(D, Bd_diff2, w1);
        alg_57_flat<ValueType>(D, Bd_diff2, w1,pD,vsize);
        // std::cout<<"57.2 w1=[";
        // for(size_t ri=0;ri<vsize;ri++) std::cout<<w1[ri]<<",";
        // std::cout<<"]"<<std::endl;
        Vectorwise_sum_helpers_flat(w1,r,vsize);
        // std::cout<<di<<"<-di,r=[";
        // for(size_t ri=0;ri<vsize;ri++) std::cout<<r[ri]<<",";
        // std::cout<<"]"<<std::endl;
        delete[] Bd_diff2;
    }
    // std::cout<<"last r=[";
    // for(size_t ri=0;ri<vsize;ri++) std::cout<<r[ri]<<",";
    // std::cout<<"]"<<std::endl;
    // return r;
}

template void alg_84_flat<float>(const size_t, const float*, 
    float*, const float*,const float*,size_t, float*);

template <typename ValueType>
void alg_84_mergeBd_flat(ValueType* Bd_diff, const ValueType* Bdiff,
     const size_t t, size_t row, size_t col, size_t dim) {
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < col; ++j) {
            Bd_diff[i*col*dim + j*dim + t] = Bdiff[i*col*dim + j*dim + t];
        }
    }
}
template void alg_84_mergeBd_flat<float>(float*,const float*,
     const size_t,size_t,size_t,size_t);


template <typename ValueType>
void alg_82_matrices(const size_t D, const std::vector<Matrixt<ValueType>>& B,
                                const std::vector<ValueType>& G, 
                                    std::vector<ValueType>& v) {
    
    runtime_check(D != B.size(), "Dimension of problem and number of shape function matrices B building tensor space not equal!");

    // size_t i_nq = B[0][0].size();
    // size_t i_nf = B[0][0][0].size();

    alg_57_matrices(D, B, v);
    Vectorwise_Mult_helpers_vector(G,v);

    std::vector<Matrixt<ValueType>> B2 = swapaxes_matrices(B);

    alg_57_matrices(D, B2, v);
    //TODO: implement also algorithm 58
        //else:
        //    w1 = alg_58(D, B, v)
        //w2 = G * w1
        //r = alg_58(D, np.swapaxes(B, 0, 1), w2)
}
template void alg_82_matrices<float>(const size_t, const std::vector<Matrixt<float>>&, const std::vector<float>&, std::vector<float>&);
template void alg_82_matrices<double>(const size_t, const std::vector<Matrixt<double>>&, const  std::vector<double>&, std::vector<double>&);

template <typename ValueType>
void alg_82_flat(const size_t D, 
                    ValueType* B,
                    ValueType* G,
                    ValueType* v, 
                    size_t pD) {
    
    // runtime_check(D != B.size(), "Dimension of problem and number of shape function matrices B building tensor space not equal!");

    // size_t i_nq = B[0][0].size();
    // size_t i_nf = B[0][0][0].size();
    size_t vsize=(pD+1)*(pD+1);
    alg_57_flat<ValueType>(D, B, v,pD,vsize);
    Vectorwise_Mult_helpers_flat(G,v,vsize);

    // std::vector<Matrixt<ValueType>> B2 = swapaxes_matrices(B);

    ValueType* B2 = swapaxes_flat(B,D,pD+1,pD+1);

    alg_57_flat<ValueType>(D, B2,v,pD,vsize);
    //TODO: implement also algorithm 58
        //else:
        //    w1 = alg_58(D, B, v)
        //w2 = G * w1
        //r = alg_58(D, np.swapaxes(B, 0, 1), w2)
}
template void alg_82_flat<float>(const size_t, float*, float*, float*,size_t);
template void alg_82_flat<double>(const size_t, double*, double*, double*,size_t);





// void alg_57_matrices(const int D, const std::vector<Matrix>& R, std::vector<double>& P) {
template <typename ValueType>
void alg_57_matrices(const size_t D, const std::vector<Matrixt<ValueType>>& R, std::vector<ValueType>& P) {

    /* here we want to create a copy, that is the reason behing not calling the vector P as
    a pointer.
    */
    // runtime_check(D != R.size(), "Dimension of problem and number of matrices R for Kronecker product not equal!");
    // runtime_check(R[0].size1() != R[0].size2(), "Algorithm just works for quadratic matrices R!");

    size_t n_i = R[0].size();
    size_t size = std::pow(n_i, D);
    std::vector<ValueType> T(P.size(), 0.0);
    size_t stride = (size / n_i);
    for (size_t i = 0; i < D; ++i) {// in range(0, D) :
        for (size_t j = 0; j < stride; ++j) {//this creates 
            std::vector<ValueType> Pfraction(P.begin() + j * n_i, P.begin() + (j + 1) * n_i);
            Matrixt<ValueType> Rfraction(n_i);
            genVV(Rfraction,n_i, n_i);
            for (size_t Ri = 0; Ri < n_i; Ri++) {
                for (size_t Rj = 0; Rj < n_i; Rj++) {
                    // std::cout<<"["<<Ri<<"]["<<Rj<<"]"<<std::endl;                
                    Rfraction[Ri][Rj] = R[Ri][Rj][i];
                }
            }
            size_t value_idx = 0;
            std::vector<ValueType> val=matvec(Rfraction,Pfraction);//Matrix*vector
            for (size_t inner = j; inner < size; inner += stride) {
                T[inner] = val[value_idx++];
            }
        }//# Fehler in Davide's Diss -> linke Seite beginnend bei j
        std::swap(P, T);
    }
}
template void alg_57_matrices<float>(const size_t, const std::vector<Matrixt<float>>&,  std::vector<float>&);
template void alg_57_matrices<double>(const size_t, const std::vector<Matrixt<double>>&,  std::vector<double>&);


template <typename ValueType>
void alg_57_flat(const size_t D, 
                    ValueType* R, 
                    ValueType* P,size_t pD, size_t p_size) {
    // '''here we want to create a copy, that is the reason behing not calling the vector P as
    // a pointer. 
    // n_i
    // R[pD+1][pD+1][2] 
    // P[(pD+1)^2]
    // '''
    // runtime_check(D != R.size(), "Dimension of problem and number of matrices R for Kronecker product not equal!");
    // runtime_check(R[0].size1() != R[0].size2(), "Algorithm just works for quadratic matrices R!");
    
    size_t s = std::pow(pD+1, D);
    ValueType* T= new ValueType[p_size];//(P.size(), 0.0);
    ValueType* Pfraction = new ValueType[pD+1];
    ValueType* Rfraction= new ValueType[(pD+1)*(pD+1)];
    //nullptr;//new ValueType*[pD+1];
    ValueType* val_T= new ValueType[pD+1];

    // genVV2(&Rfraction,pD+1,pD+1);//initialize matrix
    std::fill(T, T + p_size, 0.0);//initialize T

    size_t stride = (s / (pD+1));
    for (size_t i = 0; i < D; ++i){
        for (size_t j = 0; j < stride; ++j){ 
            getPfraction(Pfraction,P,j,p_size,pD+1);
            getRfraction(Rfraction,R,D,i,pD+1);
            size_t value_idx = 0;
            matvec2(val_T,Rfraction,Pfraction, pD+1,pD+1);//Matrix*vector
            for (size_t inner = j; inner < s; inner += stride) {
                T[inner] = val_T[value_idx++];
            }
        }//# Fehler in Davide's Diss -> linke Seite beginnend bei j
        std::swap(P, T);
    }
    delete[] Pfraction;
    delete[] Rfraction; 
    delete[] T;
    delete[] val_T;
}
template void alg_57_flat<float>(const size_t, float*, float*,size_t,size_t);
template void alg_57_flat<double>(const size_t, double*, double*,size_t,size_t);
} // namespace reference

// The Vectorwise_* helpers live in otFlyGPU (cubase.hpp), so they are instantiated outside reference.
template void Vectorwise_sum_helpers_flat<float>(const float* R,
    float* s,size_t ssize);
template void Vectorwise_Mult_helpers_vector<double>(const std::vector<double>&, std::vector<double>&);
template void Vectorwise_Mult_helpers_vector<float>(const std::vector<float>&, std::vector<float>&);
template void Vectorwise_Mult_helpers_flat<double>(const double*, double*,size_t);
template void Vectorwise_Mult_helpers_flat<float>(const float*, float*,size_t);


template <typename ValueType>
std::vector<ValueType> kroncu(const std::vector<double>& a, const std::vector<double>& b, ValueType factor) {
    /*
    * 1D kronecker product.
        * Input: Vector, Vector, factor
        * Output: Vector
    */
    size_t m = a.size();
    size_t p = b.size();
    std::vector<ValueType> out(m * p,0.0);

    for (size_t i = 0; i < m; ++i) {
            for (size_t k = 0; k < p; ++k) {
                    out[i*p+k] = a[i] * b[k] * factor;
                }
    }
    return out;
}

template std::vector<float>  kroncu<float>(const std::vector<double>&, const std::vector<double>&, float);
template std::vector<double>  kroncu<double>(const std::vector<double>&, const std::vector<double>&, double);


template <typename ValueType>
void kron_flat(ValueType* G,const std::vector<double>& a,
     const std::vector<double>& b, 
     ValueType factor,size_t m) {
    /*
    * 1D kronecker product.
        * Input: Vector, Vector, factor
        * Output: Vector
    */
    // size_t m = a.size();
    // size_t p = b.size();
    // ValueType* out = new ValueType[m * p]; // Allocate memory for the output array

    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < m; ++k) {
            G[i * m + k] = a[i] * b[k] * factor;
        }
    }
    // return out;
}

template void kron_flat<float>(float*,const std::vector<double>&, const std::vector<double>&, float,size_t);
template void kron_flat<double>(double*,const std::vector<double>&, const std::vector<double>&, double,size_t);

std::vector<size_t> arangecu(size_t number,size_t addition) {
    std::vector<size_t> outV;
    for (size_t i = 0; i < number; ++i) {
        outV.push_back(i+addition);
    }
    return outV;
}


namespace reference {
std::vector<size_t> createLocationMapcu(size_t polynomialDegree, 
    size_t ndofdirection1, 
    size_t ielement, 
    size_t jelement){ 

    std::vector<size_t> dofIndicesI = arangecu(polynomialDegree + 1, polynomialDegree * ielement);
    std::vector<size_t> dofIndicesJ = arangecu(polynomialDegree + 1, polynomialDegree * jelement);
    std::vector<size_t> LocationMap;

    for (size_t i = 0; i < polynomialDegree+1; ++i) {
        for (size_t j = 0; j < polynomialDegree+1; ++j) {
            LocationMap.push_back(dofIndicesI[i]*ndofdirection1 + dofIndicesJ[j]);
        }
    }
    return LocationMap;
}

std::vector<std::vector<size_t>> createLocationMapVector(Params params){
    //this will create a nx*ny vector. Each line will contain a ValueType
    // array of the 
    std::vector<std::vector<size_t>> LocMapVector;//= new std::vector<std::vector<ValueType>>;
    LocMapVector.reserve(params.nelements(0) * params.nelements(1));
    for (size_t ielement = 0; ielement < params.nelements(0); ++ielement) {
        for (size_t jelement = 0; jelement < params.nelements(1); ++jelement) {

            LocMapVector.push_back(createLocationMapcu(params.polynomialDegree(),
            params.ndofdirection()[1],
            ielement, jelement));
        }
    }
    return LocMapVector;
}
} // namespace reference


template <typename ValueType>
void genVV(Matrixt<ValueType>& Vec, size_t n, size_t m){
    std::vector<ValueType> buffer(n * m, 0); // Single allocation
    for (size_t i = 0; i < n; i++)
        Vec[i] = std::vector<ValueType>(&buffer[i * m], &buffer[(i + 1) * m]); 
    // Point to the buffer
}
template void genVV<float>(Matrixt<float>&, size_t, size_t);
template void genVV<double>(Matrixt<double>&, size_t, size_t);

template <typename ValueType>
void genVV2(ValueType*** matrix, size_t n, size_t m){
    // '''
    // This function is mainly to initialize a 
    // Matrix of pointers size n. You could assign values later
    // '''
    
    (*matrix) = new ValueType*[n];

    (*matrix)[0] = new ValueType[n * m];

    for (size_t i = 1; i < n; i++) {
        (*matrix)[i] = (*matrix)[0] + i * m;
    }
}
template void genVV2<float>(float***, size_t, size_t);
template void genVV2<double>(double***, size_t,size_t);


template <typename ValueType>
std::vector<ValueType> matvec(Matrixt<ValueType>& Mat, 
                                std::vector<ValueType>& vector){
    std::vector<ValueType> result(vector.size(), 0.0);
    for (size_t iRow = 0; iRow < Mat.size(); ++iRow){
        for (size_t jCol = 0; jCol < Mat[0].size(); ++jCol){
            result[iRow] += Mat[ iRow ] [jCol] * vector[jCol];
        }
    }
    return result;
}
template std::vector<float> matvec<float>(Matrixt<float>&, std::vector<float>&);
template std::vector<double> matvec<double>(Matrixt<double>&, std::vector<double>&);

template <typename ValueType>
void matvec2(ValueType* result, ValueType* Mat, ValueType* vector, size_t r, size_t c){
    for (size_t iRow = 0; iRow < r; ++iRow){
        result[iRow] = 0.0;
        for (size_t jCol = 0; jCol < c; ++jCol){
            result[iRow] += Mat[iRow*c+jCol] * vector[jCol];
        }
    }
}
template void matvec2<float>(float*,float*, float*,size_t,size_t);
template void matvec2<double>(double*,double*, double*,size_t,size_t);

namespace reference {
template <typename ValueType>
void swapaxes_matrices(std::vector<Matrixt<ValueType>>& A) {
    //  trying to reproduce np.swapaxes(A,0,1);
    // this is only if t and i is the same number!!!
    //     meaning this only works for algorithm57
    runtime_check(A.size() == A[0].size(),
        "Dimension problem and number of Matrices not equal to number of rows!");

    std::vector<Matrixt<ValueType>> Mout(A);//VectorMatrix(A.size(), A[0].size1(), A[0].size2());

    for (size_t t = 0; t < A.size(); ++t) {
        for (size_t i = 0; i < A[0].size(); ++i) {
            for (size_t j = 0; j < A[0][0].size(); ++j) {
                A[t][i][j] = Mout[i][t][j];
            }
        }
    }
}
template void swapaxes_matrices<float>(std::vector<Matrixt<float>>&);
template void swapaxes_matrices<double>(std::vector<Matrixt<double>>&);

template <typename ValueType>
ValueType* swapaxes_flat(const ValueType* B,size_t dim, size_t row, size_t col) {
    //  trying to reproduce np.swapaxes(A,0,1);
    // this is only if t and i is the same number!!!
    //     meaning this only works for algorithm57
    // runtime_check(A.size() == A[0].size(),
        // "Dimension problem and number of Matrices not equal to number of rows!");
    ValueType* A = new ValueType[col*row*dim];
    std::copy(B, B + col * row * dim, A);
    for (size_t t = 0; t < col; ++t) {
        for (size_t i = t+1; i < row; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                std::swap(A[(t * row + i) * dim + j], A[(i * row + t) * dim + j]);
            }
        }
    }
    return A;
}
template float* swapaxes_flat<float>(const float*,size_t,size_t,size_t);
template double* swapaxes_flat<double>(const double*,size_t,size_t,size_t);

template <typename ValueType>
std::vector<Matrixt<ValueType>> swapaxes_matrices(const std::vector<Matrixt<ValueType>>& A) {
    /* trying to reproduce np.swapaxes(A,0,1);
    this is only if t and i is the same number!!!
        meaning this only works for algorithm57
    */

    // runtime_check(A.size() == A[0].size1(),
    //     "Dimension problem and number of Matrices not equal to number of rows!");

    // = VectorMatrix(A.size(), A[0].size(), A[0][0].size());
    std::vector<Matrixt<ValueType>> Mout(A.size(),
                std::vector<std::vector<ValueType>>(A[0].size(), 
                                std::vector<ValueType>(A[0][0].size())));

    for (size_t t = 0; t < A.size(); ++t) {
        for (size_t i = 0; i < A[0].size(); ++i) {
            for (size_t j = 0; j < A[0][0].size(); ++j) {
                Mout[t][i][j] = A[i][t][j];
            }
        }
    }
    return Mout;
}
template std::vector<Matrixt<float>> swapaxes_matrices<float>(const std::vector<Matrixt<float>>&);
template std::vector<Matrixt<double>> swapaxes_matrices<double>(const std::vector<Matrixt<double>>&);
} // namespace reference

template <typename ValueType>
std::vector<Matrixt<ValueType>> concatenate_matrices(const Matrixt<ValueType>& A, 
                                    const Matrixt<ValueType>& C) {
    /*Trying to reproduce np.concatenate(A,C,2) from
        numpy library. 
    */
    std::vector<Matrixt<ValueType>> B;
    size_t rows = A.size();
    size_t cols = A[0].size();
    for (size_t c = 0; c < cols; ++c) {
        Matrixt<ValueType> Binner(rows, std::vector<ValueType>(rows,2.0));

        for (size_t r = 0; r < rows; ++r) {
            Binner[r][0] = A[c][r];
            Binner[r][1] = C[c][r];
        }
        B.push_back(Binner);
    }
    return B;
}
template std::vector<Matrixt<float>> concatenate_matrices<float>(const Matrixt<float>&,const Matrixt<float>&);
template std::vector<Matrixt<double>> concatenate_matrices<double>(const Matrixt<double>&,const Matrixt<double>&);

template <typename ValueType>
void concatenate_flat(ValueType* B, const Matrixt<ValueType>& A, 
                                    const Matrixt<ValueType>& C, 
                                    size_t rows, size_t dim) {
    /*Trying to reproduce np.concatenate(A,C,2) from
        numpy library. 
        rows = pD+1;
        cols = pD+1;
    */
    for (size_t c = 0; c < rows; ++c) {
        for (size_t i = 0; i < rows; ++i) {
            B[c *rows*dim + i * dim + 0] = A[c][i];
            B[c *rows*dim + i * dim + 1] = C[c][i];
        }
    }
}

template void concatenate_flat<float>(float*,const Matrixt<float>&,const Matrixt<float>&,size_t,size_t);
template void concatenate_flat<double>(double*,const Matrixt<double>&,const Matrixt<double>&,size_t,size_t);

template <typename ValueType>
Matrixt<ValueType> get_ShapesDiffcu(const std::vector<double>& coordinates,const size_t diff){
    adhocpp::numeric::LagrangeBasisFunctions shapes0(coordinates);
    Matrixt<ValueType> shapesDiff0;//(coordinates.size(), std::vector<ValueType>(coordinates.size(),val));

    for (size_t coor = 0; coor < coordinates.size(); ++coor) {
        std::vector<ValueType> shapesDiff0_r;
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
template Matrixt<float> get_ShapesDiffcu<float>(const std::vector<double>&, const size_t);
template Matrixt<double> get_ShapesDiffcu<double>(const std::vector<double>&, const size_t);



template <typename ValueType>
void ShapeFunctionFactor(Matrixt<ValueType>& M, const ValueType f) {
    for(size_t r = 0;r<M.size();r++){
        for(size_t c = 0;c<M[0].size();c++){
            M[r][c] = M[r][c] * f;
        }
    }
}
template void ShapeFunctionFactor<float>(Matrixt<float>&, const float);
template void ShapeFunctionFactor<double>(Matrixt<double>&, const double);










template <typename ValueType>
void hellocudabase(ValueType num){
    //example of how to use different kind of value types
    //this is relevant when we actually have a analysis for 
    //diferent kind of variables inside our cuda implementation. 
    std::cout<<"Hello from the other side"<<num<<std::endl;
}
template void hellocudabase<float>(float);
template void hellocudabase<int>(int);
template void hellocudabase<double>(double);



// #define DECLARE_ShapeFunctionFactor(_vtype)                                    \
//     void ShapeFunctionFactor(Matrixt<ValueType>& M, const ValueType f)
// INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_MATMUL);

}//otFly

}//cie





