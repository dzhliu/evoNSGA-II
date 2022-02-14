#ifndef ALPHADOMINANCEHANDLER_H
#define ALPHADOMINANCEHANDLER_H

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

#include "GPGOMEA/Evolution/NSGA2GenerationHandler.h"  

class AlphaDominance : public NSGA2GenerationHandler{

public:

AlphaDominance(ConfigurationOptions * conf, TreeInitializer * tree_initializer, Fitness * fitness, SemanticLibrary * semlib = NULL, SemanticBackpropagator * semback = NULL) : 
    NSGA2GenerationHandler(conf, tree_initializer, fitness, semlib,semback){
    };

    
    virtual void PerformGeneration(std::vector<Node *> & population) override;
    
    void PerformGeneration_alpha_dominance(std::vector<Node *> & population);
    void PerformGeneration_adaptive_alpha_dominance(std::vector<Node *> & population);

    std::vector<std::vector<Node*>> FastNonDominatedSorting_alpha_dominance(std::vector<Node*> & population);
    std::vector<std::vector<Node*>> FastNonDominatedSorting_adaptive_alpha_dominance(std::vector<Node*> & population);

    void get_alpha();//for alpha dominance 

    std::vector<double> dominance_alpha;

    double u_eff;
    double l_eff;
    int u_size;
    int l_size;
    double current_alpha;
    void adaptive_update_alpha(std::vector<Node *> & population);//for adaptive alpha dominance

private:

};



#endif /* GENERATIONHANDLER_H */