#ifndef SPEA2GENERATIONHANDLER_H
#define SPEA2GENERATIONHANDLER_H

#include "GPGOMEA/Evolution/GenerationHandler.h"  
#include "GPGOMEA/Evolution/NSGA2GenerationHandler.h"  
#include "GPGOMEA/Genotype/Node.h"
#include "GPGOMEA/Utils/ConfigurationOptions.h"
#include "GPGOMEA/Selection/TournamentSelection.h"
#include "GPGOMEA/Variation/SubtreeVariator.h"
#include "GPGOMEA/Semantics/SemanticBackpropagator.h"
#include "GPGOMEA/Semantics/SemanticLibrary.h"

#include <armadillo>
#include <vector>
#include <unordered_map>

class SPEA2GenerationHandler : public NSGA2GenerationHandler {
public:


    SPEA2GenerationHandler(ConfigurationOptions * conf, TreeInitializer * tree_initializer, Fitness * fitness, SemanticLibrary * semlib = NULL, SemanticBackpropagator * semback = NULL) : 
        NSGA2GenerationHandler(conf, tree_initializer, fitness, semlib, semback) {}; // inherit same constructor


    virtual void PerformGeneration(std::vector<Node *> & population) override;

    void GetStrength(std::vector<Node *> & population);

    std::vector<Node*> archive;//this is not the archive from start to end of the whole evolution process, but just used for SPEA2

    std::vector<Node *> archive_output; //store all non_dominated solutions find so far since the 1st generation, this is what we should output for quality evaluation

    void SPEA2EnvironmentSelection(std::vector<Node *> & population);

    double calculate_distance(Node * n1, Node * n2);

    std::vector<std::vector<double>> generate_update_SPEA2_crowding_distance_table();

    Node * SPEA2TournamentSelectionWinner(const std::vector<Node*>& candidates, size_t tournament_size, std::vector<std::vector<double>> dist_table);

    void save_final_population(std::vector<Node*> & population);

    void update_archive(std::vector<Node *> & population);

private:

};

#endif /* GENERATIONHANDLER_H */

