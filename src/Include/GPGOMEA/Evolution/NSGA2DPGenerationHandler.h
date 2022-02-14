#ifndef NSGA2DPGENERATIONHANDLERHANDLER_H
#define NSGA2DPGENERATIONHANDLERHANDLER_H


#include "GPGOMEA/Evolution/EvolutionState.h"
#include "GPGOMEA/Evolution/PopulationInitializer.h"
#include "GPGOMEA/Utils/Logger.h"
#include "GPGOMEA/GOMEA/GOMEAGenerationHandler.h"
#include "GPGOMEA/Evolution/GenerationHandler.h"
#include "GPGOMEA/Fitness/MOFitness.h"


#include "GPGOMEA/Evolution/NSGA2GenerationHandler.h" 


class NSGA2DPGenerationHandler : public NSGA2GenerationHandler {
    
public:
    NSGA2DPGenerationHandler(ConfigurationOptions * conf, TreeInitializer * tree_initializer, Fitness * fitness, SemanticLibrary * semlib = NULL, SemanticBackpropagator * semback = NULL) : 
    NSGA2GenerationHandler(conf, tree_initializer, fitness, semlib,semback){};

    
    virtual void PerformGeneration(std::vector<Node *> & population) override;

private:

};


#endif /* GENERATIONHANDLER_H */