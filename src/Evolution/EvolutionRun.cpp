/*
 


 */

/* 
 * File:   EvolutionRun.cpp
 * Author: virgolin
 * 
 * Created on June 28, 2018, 11:28 AM
 */

#include "GPGOMEA/Evolution/EvolutionRun.h"

#include <malloc.h>

using namespace std;
using namespace arma;

void EvolutionRun::Initialize() {

    elitist = NULL;

    // Initialize population
    population = PopulationInitializer::InitializeTreePopulation(*config, *tree_initializer, *fitness);

    // Compute fitness of the population
    pop_fitnesses = fitness->GetPopulationFitness(population, true, config->caching);

    // create semantic library if needed
    if (config->semantic_variation && config->semback_library_type == SemanticLibraryType::SemLibRandomStatic)
        semantic_library->GenerateRandomLibrary(config->semback_library_max_height, config->semback_library_max_size, *fitness, config->functions, config->terminals, *tree_initializer, config->caching);
}

void EvolutionRun::DoGeneration() {
    // create semantic library if needed
    if (config->semantic_variation) {
        if (config->semback_library_type == SemanticLibraryType::SemLibRandomDynamic)
            semantic_library->GenerateRandomLibrary(config->semback_library_max_height, config->semback_library_max_size, *fitness, config->functions, config->terminals, *tree_initializer, config->caching);
        else if (config->semback_library_type == SemanticLibraryType::SemLibPopulation)
            semantic_library->GeneratePopulationLibrary(config->semback_library_max_height, config->semback_library_max_size, population, *fitness, config->caching);
    }
    // perform generation
    std::cout<<"execute PerformGeneration:generation_handler->PerformGeneration(population)"<<std::endl;

    std::cout<<"start to execute malloc_trim(0)...";
    malloc_trim(0); 
    std::cout<<"finished!"<<std::endl;

    generation_handler->PerformGeneration(population);
   
    unsigned int sum_exp_len = 0;
    for(auto n : population){
        sum_exp_len += n->GetSubtreeNodes(true).size();
    }
    std::cout<<"after this iteration, the sum number of exp length in population is:"<<sum_exp_len <<std::endl;
    
    std::cout<<"finish one iteration!"<<std::endl;
    // update stats
    pop_fitnesses = fitness->GetPopulationFitness(population, false, config->caching);
    Node * best = population[ index_min(pop_fitnesses) ]; 
    size_t best_size = best->GetSubtreeNodes(true).size();
    double_t best_fit = best->cached_fitness;
    
    if (best_fit < elitist_fit) {
        elitist_fit = best_fit;
        if (elitist)
            elitist->ClearSubtree();
        elitist = best->CloneSubtree();
        elitist_size = best_size;
    }
    // update mo_archive
    if (is_multiobj){
        // DEPRECATED: For each solution in the population with best rank, try to fit it in the archive
        for(Node * solution : population) {
            if (solution->rank != 0)
                continue;
            // check if worth inserting in the archive_size
            bool solution_is_dominated = false;
            bool identical_objectives_already_exist;
            for(size_t i = 0; i < mo_archive.size(); i++) {
                // check domination
                Node * n = mo_archive[i];
                solution_is_dominated = n->Dominates(solution);
                if (solution_is_dominated)
                    break;

                identical_objectives_already_exist = true;
                for(size_t j = 0; j < solution->cached_objectives.n_elem; j++){
                    if (solution->cached_objectives[j] != n->cached_objectives[j]) {
                        identical_objectives_already_exist = false;
                        break;
                    }
                }
                if (identical_objectives_already_exist)
                    break;

                bool n_is_dominated = solution->Dominates(n);
                if (n_is_dominated) {
                    n->ClearSubtree();
                    mo_archive[i] = NULL;  // keep this guy 
                }
            }

            if (!solution_is_dominated && !identical_objectives_already_exist) {
                mo_archive.push_back(solution->CloneSubtree());    // clone it
            }

            vector<Node*> updated_archive; updated_archive.reserve(mo_archive.size());
            for(size_t i = 0; i < mo_archive.size(); i++)
                if (mo_archive[i])
                    updated_archive.push_back(mo_archive[i]);
            mo_archive = updated_archive;
        }
    }

    // TODO: log what you like


}
