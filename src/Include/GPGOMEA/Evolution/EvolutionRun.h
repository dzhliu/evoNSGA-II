/*
 


 */

/* 
 * File:   EvolutionRun.h
 * Author: virgolin
 *
 * Created on June 28, 2018, 11:28 AM
 */

#ifndef EVOLUTIONRUN_H
#define EVOLUTIONRUN_H


#include "GPGOMEA/Evolution/EvolutionState.h"
#include "GPGOMEA/Evolution/PopulationInitializer.h"
#include "GPGOMEA/Utils/Logger.h"
#include "GPGOMEA/GOMEA/GOMEAGenerationHandler.h"
#include "GPGOMEA/Evolution/GenerationHandler.h"
#include "GPGOMEA/Evolution/NSGA2GenerationHandler.h"
#include "GPGOMEA/Evolution/NSGA2DPGenerationHandler.h"
#include "GPGOMEA/Evolution/SPEA2GenerationHandler.h"
#include "GPGOMEA/Fitness/MOFitness.h"

#include "GPGOMEA/Evolution/AlphaDominance.h"

#include <iostream>
#include <vector>

class EvolutionRun {
public:

    EvolutionRun(EvolutionState & st) {
        config = new ConfigurationOptions(*st.config); // clone of the configuration settings
        if (st.semantic_library)
            semantic_library = new SemanticLibrary(*st.semantic_library); // clone semantic library

        if (st.config->gomea) {
            generation_handler = new GOMEAGenerationHandler(*((GOMEAGenerationHandler*) st.generation_handler)); // clone generation handler
            if (((GOMEAGenerationHandler*) st.generation_handler)->linkage_normalization_matrix)
                ((GOMEAGenerationHandler*) generation_handler)->linkage_normalization_matrix = new arma::mat(); // detach pointer to previous linkage normalization matrix
            ((GOMEAGenerationHandler*) generation_handler)->gomea_converged = false;
        } else {
            if (dynamic_cast<MOFitness*>(st.fitness)){
               
                if(st.config->algorithm_framework == "NSGA2")
                {
                    if(st.config->methodtype.compare("alpha_dominance") == 0 || st.config->methodtype.compare("adaptive_alpha_dominance") == 0)
                    {
                        generation_handler = (GenerationHandler *)new AlphaDominance(st.config,st.tree_initializer, st.fitness,st.semantic_library,st.semantic_backprop);
                    }
                    else
                    {
                        std::cout<<"Create NSGA2GenerationHandler"<<std::endl;
                        //generation_handler = new NSGA2GenerationHandler(*dynamic_cast<NSGA2GenerationHandler*>(st.generation_handler));
                        generation_handler = (GenerationHandler *)new NSGA2GenerationHandler(st.config,st.tree_initializer, st.fitness,st.semantic_library,st.semantic_backprop);                    
                    }
                }
                else if(st.config->algorithm_framework == "NSGA2DP")
                {
                    generation_handler = (GenerationHandler *)new NSGA2DPGenerationHandler(st.config,st.tree_initializer, st.fitness,st.semantic_library,st.semantic_backprop);
                }
                else if(st.config->algorithm_framework == "SPEA2")
                {
                    std::cout<<"Create SPEA2GenerationHandler!!!"<<std::endl;
                    SPEA2GenerationHandler * spea2h = new SPEA2GenerationHandler(st.config,st.tree_initializer, st.fitness,st.semantic_library,st.semantic_backprop);
                    generation_handler = (GenerationHandler *)spea2h;
                }
                //else if(st.config->algorithm_framework == "LengthControlTruncation")
                else if(st.config->algorithm_framework == "evoNSGA2")
                {
                    LengthControlTruncationGenerationHandler * lcth = new LengthControlTruncationGenerationHandler(st.config,st.tree_initializer, st.fitness,st.semantic_library,st.semantic_backprop);
                    generation_handler = (GenerationHandler *) lcth;
                }
                else
                {
                    std::cout<<"error algorithm_framework definition error! The algorithm framework is:"<<st.config->algorithm_framework<<std::endl;
                    exit(-1);
                }
                is_multiobj = true;
            }
            else
                generation_handler = new GenerationHandler(*st.generation_handler);
            if (st.semantic_library)
                generation_handler->semlib = semantic_library;
        }
     
        // set correct handler to config
        generation_handler->conf = config;

        tree_initializer = st.tree_initializer; // share same tree initializer
        fitness = st.fitness; // share same fitness
    };

    virtual ~EvolutionRun() {
        for (Node * n : population) {
            n->ClearSubtree();
        }

        if (elitist)
            elitist->ClearSubtree();

        delete config;
        if (semantic_library)
            delete semantic_library;
        delete generation_handler;

        //fix bug: memory leak 20211105
        for(Node * n: mo_archive){
            n->ClearSubtree();
            n = nullptr;
        }
        mo_archive.clear();
        //fix bug finished: memory leak 20211105

    }

    void Initialize();
    void DoGeneration();

    std::vector<Node*> population;
    ConfigurationOptions * config = NULL;
    TreeInitializer * tree_initializer = NULL;
    GenerationHandler * generation_handler = NULL;
    Fitness * fitness = NULL;
    SemanticLibrary * semantic_library = NULL;

    arma::vec pop_fitnesses;

    Node * elitist = NULL;
    double_t elitist_fit = arma::datum::inf;
    size_t elitist_size;

    bool is_multiobj = false;
    std::vector<Node*> mo_archive;

private:

};

#endif /* EVOLUTIONRUN_H */

