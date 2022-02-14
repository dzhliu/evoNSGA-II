/*
 


 */

/* 
 * File:   SymbolicRegressionLinearScalingFitness.h
 * Author: virgolin
 *
 * Created on June 27, 2018, 7:23 PM
 */

#ifndef SYMBOLICREGRESSIONLINEARSCALINGFITNESS_H
#define SYMBOLICREGRESSIONLINEARSCALINGFITNESS_H

#include "GPGOMEA/Fitness/SymbolicRegressionFitness.h"
#include "GPGOMEA/Utils/Utils.h"

#include <armadillo>
#include <math.h>

class SymbolicRegressionLinearScalingFitness : public SymbolicRegressionFitness {
public:

    double_t ComputeFitness(Node* n, bool use_caching) override;

    double_t GetTestFit(Node * n) override;
    
    double_t GetValidationFit(Node* n) override;

    double_t ComputeLinearScalingMSE(const arma::vec & P);
    
    double_t ComputeLinearScalingMSE(const arma::vec & P, const arma::vec & Y, double_t a, double_t b);

    std::pair<double_t, std::vector<double>> ComputeLinearScalingMSE_withsemanticvector(const arma::vec& P);

private:


};

#endif /* SYMBOLICREGRESSIONLINEARSCALINGFITNESS_H */

