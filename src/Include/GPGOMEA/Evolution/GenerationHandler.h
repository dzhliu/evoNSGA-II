/*
 


 */

/* 
 * File:   GenerationHandler.h
 * Author: virgolin
 *
 * Created on June 28, 2018, 12:08 PM
 */

#ifndef GENERATIONHANDLER_H
#define GENERATIONHANDLER_H

#include "GPGOMEA/Genotype/Node.h"
#include "GPGOMEA/Utils/ConfigurationOptions.h"
#include "GPGOMEA/Selection/TournamentSelection.h"
#include "GPGOMEA/Variation/SubtreeVariator.h"
#include "GPGOMEA/Semantics/SemanticBackpropagator.h"
#include "GPGOMEA/Semantics/SemanticLibrary.h"

#include <armadillo>
#include <vector>

class GenerationHandler {
public:

    GenerationHandler(ConfigurationOptions * conf, TreeInitializer * tree_initializer, Fitness * fitness, SemanticLibrary * semlib = NULL, SemanticBackpropagator * semback = NULL) {
        this->conf = conf;
        this->tree_initializer = tree_initializer;
        this->fitness = fitness;
        this->semlib = semlib;
        this->semback = semback;
    };

    virtual void PerformGeneration(std::vector<Node *> & population);
 
    virtual bool CheckPopulationConverged(const std::vector<Node*>& population);

    std::vector<Node*> MakeOffspring(const std::vector<Node *> & population, const std::vector<Node*> & parents);
    
    std::vector<Node*> MakeOffspring_exempt_elitism(const std::vector<Node *> & population, const std::vector<Node*> & parents, int num_elites);
    
    std::vector<Node*> MakeOffspring_counting_improvment_frequency(const std::vector<Node *> & population, const std::vector<Node*> & parents, std::vector<double> & mutation_ExpLength_FitnessAttempt_frequency, std::vector<std::vector<double>> & crossover_ExpLength_FitnessAttempt_frequency);

    std::vector<Node*> MakeOffspring_counting_improvment_frequency_with_ExpLenRestriction(const std::vector<Node *> & population,  const std::vector<Node*> & candidate1_list_crossover, const std::vector<Node *> candidate2_list_crossover, const std::vector<Node*> & candidate_list_mutation,  std::vector<double> mutation_ExpLength_FitnessAttempt_frequency, std::vector<std::vector<double>> crossover_ExpLength_FitnessAttempt_frequency);

    void MakeOffspring(const std::vector<Node *> & population, const std::vector<Node*> & selected_parents, std::vector<Node*>& offspring);


    bool ValidateOffspring(Node * offspring, int max_height, int max_size);

    ConfigurationOptions * conf;
    TreeInitializer * tree_initializer;
    Fitness * fitness;
    SemanticLibrary * semlib;
    SemanticBackpropagator * semback;

    //for LengthControlTruncation
    std::vector<Node*> MakeOffspring_counting_improvment_frequency_with_LCTrunction(const std::vector<Node *> & population, const std::vector<Node *> & selected_parents, std::vector<double> & crossover_attempts, std::vector<double> & mutation_attempts);

   

};

#endif /* GENERATIONHANDLER_H */

