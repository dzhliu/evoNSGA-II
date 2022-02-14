/*
 


 */


#include "GPGOMEA/Fitness/SymbolicRegressionLinearScalingFitness.h"

using namespace std;
using namespace arma;

//Dazhuang revised 20211102
std::pair<double_t, std::vector<double>> SymbolicRegressionLinearScalingFitness::ComputeLinearScalingMSE_withsemanticvector(const arma::vec& P)
{
    pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(P, TrainY, &trainY_mean, &var_comp_trainY);
    
    arma::vec res;
    if (ab.second != 0) {
        res = TrainY - (ab.first + ab.second * P);
    } else {
        res = TrainY - ab.first; // do not need to multiply by P elements by 0
    }
    double_t ls_mse = arma::mean( arma::square(res) );
    
    std::pair<double, std::vector<double>> p;
    std::vector<double> res_to_vector;
    for(int i = 0; i < res.size(); i++)
    {
        res_to_vector.push_back(res[i]);
    }
    
    p.first = ls_mse;
    p.second = res_to_vector;

    return p;
}


double_t SymbolicRegressionLinearScalingFitness::ComputeLinearScalingMSE(const arma::vec& P) {

    pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(P, TrainY, &trainY_mean, &var_comp_trainY);
    
    arma::vec res;
    if (ab.second != 0) {
        res = TrainY - (ab.first + ab.second * P);
    } else {
        res = TrainY - ab.first; // do not need to multiply by P elements by 0
    }
    
    double_t ls_mse = arma::mean( arma::square(res) );
    return ls_mse;

}

double_t SymbolicRegressionLinearScalingFitness::ComputeLinearScalingMSE(const arma::vec& P, const arma::vec& Y, double_t a, double_t b) {

    arma::vec res = Y - (a + b * P);
    double_t ls_mse = arma::sum(arma::square(res)) / res.n_elem;
    return ls_mse;

}

double_t SymbolicRegressionLinearScalingFitness::GetTestFit(Node * n) {
    vec P = n->GetOutput( TrainX, false );
    pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(P, TrainY, &trainY_mean, &var_comp_trainY);
    P = n->GetOutput(TestX, false);
    double_t fit = ComputeLinearScalingMSE(P, TestY, ab.first, ab.second);
    
    //fit = 1 - fit / as_scalar(var(TestY));
    return fit;
}

double_t SymbolicRegressionLinearScalingFitness::GetValidationFit(Node* n) {
    vec P = n->GetOutput( TrainX, false );
    pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(P, TrainY, &trainY_mean, &var_comp_trainY);
    P = n->GetOutput(ValidationX, false);
    double_t fit = ComputeLinearScalingMSE(P, ValidationY, ab.first, ab.second);
    
    //fit = 1 - fit / as_scalar(var(TestY));
    return fit;
}


double_t SymbolicRegressionLinearScalingFitness::ComputeFitness(Node* n, bool use_caching) {
    
    evaluations++;
    
    arma::vec P = n->GetOutput(TrainX, use_caching);
    
    std::pair<double,std::vector<double>> p_return =  ComputeLinearScalingMSE_withsemanticvector(P);
    
    //double_t fit = ComputeLinearScalingMSE(P);
    double_t fit = p_return.first;
    

    if (std::isnan(fit))
            fit = arma::datum::inf;
    n->cached_fitness = fit;

    // scaled output is
    //pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(P, TrainY, &trainY_mean, &var_comp_trainY);
    //arma::vec scaled_out = (ab.first + ab.second * P);
    pair<double_t,double_t> ab = Utils::ComputeLinearScalingTerms(P, TrainY, &trainY_mean, &var_comp_trainY);
    arma::vec scaled_out = (ab.first + ab.second * P);


    // n->semantic_description = scaled_out; //set<vector<double_t>> (this works) // then you need to convert arma::vec to vector<double_t> 

    /*
    vector<double_t> semantic_description; semantic_description.reserve(scaled_out.n_elem);
    for(size_t i = 0; i < scaled_out.n_elem; i++) {
        semantic_description.push_back(scaled_out[i]);
    }
    */

   n->semantic_description = p_return.second;

   return fit;
    
}
