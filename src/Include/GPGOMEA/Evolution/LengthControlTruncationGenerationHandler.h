#ifndef LengthControlTruncationGENERATIONHANDLER_H
#define LengthControlTruncationGENERATIONHANDLER_H

#include "GPGOMEA/Evolution/GenerationHandler.h"  
#include "GPGOMEA/Genotype/Node.h"
#include "GPGOMEA/Utils/ConfigurationOptions.h"
#include "GPGOMEA/Selection/TournamentSelection.h"
#include "GPGOMEA/Variation/SubtreeVariator.h"
#include "GPGOMEA/Semantics/SemanticBackpropagator.h"
#include "GPGOMEA/Semantics/SemanticLibrary.h"

#include <armadillo>
#include <vector>
#include <unordered_map>

class LengthControlTruncationGenerationHandler : public GenerationHandler {
    
public:

    LengthControlTruncationGenerationHandler(ConfigurationOptions * conf, TreeInitializer * tree_initializer, Fitness * fitness, SemanticLibrary * semlib = NULL, SemanticBackpropagator * semback = NULL) : 
        GenerationHandler(conf, tree_initializer, fitness, semlib, semback) {    
            std::cout<<"LengthControlTruncationGenerationHandler constructor...";
            crossover_attempts.assign(conf->maximum_solution_size,0);
            crossover_succeed.assign(conf->maximum_solution_size,0);
            crossover_probability.assign(conf->maximum_solution_size,0);
            mutation_attempts.assign(conf->maximum_solution_size,0);
            mutation_succeed.assign(conf->maximum_solution_size,0);
            mutation_probability.assign(conf->maximum_solution_size,0);
            std::cout<<"finished"<<std::endl;
        }; // inherit same constructor

    virtual void PerformGeneration(std::vector<Node *> & population) override;

    std::vector<std::vector<Node*>> FastNonDominatedSorting(const std::vector<Node*> & population);
    void ComputeCrowndingDistance(std::vector<Node *> & front);

    double calculate_distance(Node * n1, Node * n2);
    
    
    void save_final_population(std::vector<Node*> & population);
    void save_population(std::vector<Node*> & population);
    

    void update_archive(std::vector<Node *> & population);

    std::vector<double> crossover_attempts;
    std::vector<double> crossover_succeed;
    std::vector<double> crossover_probability;
    std::vector<double> mutation_attempts;
    std::vector<double> mutation_succeed;
    std::vector<double> mutation_probability;


    std::vector<Node*> archive;

    void update_probability(const std::vector<Node *> & offspring, const std::vector<Node *> & population, double middle_accuracy, bool use_interpol);

    std::vector<int> quantity_individuals_ExpLen_in_population;//based on the probibality of crossover and mutation, calculate the maximum number of individuals than is allowed to exist in the next generation
    std::map<size_t,size_t> size_limits_for_selection;

    std::vector<Node *> survivors_selection(std::vector<Node *> & population, size_t survivor_size);
    std::vector<Node *> survivors_selection_old(std::vector<Node *> & population,std::vector<Node *> & selection);

    double calculate_middle_accuracy(std::vector<Node *> & population);

    bool use_interpol = false;


};

#endif /* GENERATIONHANDLER_H */