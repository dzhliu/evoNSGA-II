/*
 


 */

/* 
 * File:   GenerationHandler.cpp
 * Author: virgolin
 * 
 * Created on June 28, 2018, 12:09 PM
 */

#include "GPGOMEA/Evolution/GenerationHandler.h"
#include "GPGOMEA/Fitness/SymbolicRegressionLinearScalingFitness.h"

using namespace std;
using namespace arma;

void GenerationHandler::PerformGeneration(std::vector<Node*> & population) {
    // Selection
    // Classic selection with replacement
    std::vector<Node *> selected_parents;
    selected_parents.reserve(population.size());
    for (size_t i = 0; i < population.size(); i++) {
        selected_parents.push_back(TournamentSelection::GetTournamentSelectionWinner(population, conf->tournament_selection_size));
    }
    // Alternative: pop-wise tournament selection that visits all population members
    //std::vector<Node *> selected_parents = TournamentSelection::PopulationWiseTournamentSelection(population, population.size(), conf->tournament_selection_size);

    // Variation
    vector<Node*> offspring = MakeOffspring(population, selected_parents);

    // Update population
    for (size_t i = 0; i < population.size(); i++) {
        population[i]->ClearSubtree();
    }
    population = offspring;

    // Update fitness
    fitness->GetPopulationFitness(population, true, conf->caching);
}





std::vector<Node*> GenerationHandler::MakeOffspring_exempt_elitism(const std::vector<Node *> & population, const std::vector<Node*> & selected_parents, int num_elites)
{
    std::cout<<" start to MakeOffspring exempt elitism...";
	std::vector<Node *> offspring(population.size()-num_elites, NULL);

    // Variation
    size_t offspring_size, variator_limit;
#pragma omp parallel
    {
        size_t offspring_size_pvt = 0;
        size_t variator_limit_pvt = (population.size()-num_elites) * conf->subtree_crossover_proportion;
        size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

        max_attempts_variation = 10;
        attempts_variation = 0;

        population_chunk = (variator_limit_pvt / omp_get_num_threads());
        population_chunk_start_idx = population_chunk * omp_get_thread_num();

        // crossover
#pragma omp for schedule(static)
        for (size_t i = 0; i < variator_limit_pvt; i += 2) {
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeCrossover(*selected_parents[i], *selected_parents[population_chunk_start_idx + randu() * population_chunk], conf->uniform_depth_variation, conf->caching);
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = selected_parents[i]->CloneSubtree();
            }
            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree();
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree();
                }
            } else
                oo.second->ClearSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

        // mutation
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->subtree_mutation_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeMutation(*selected_parents[i], *tree_initializer, conf->functions, conf->terminals, conf->initial_maximum_tree_height, conf->uniform_depth_variation, conf->caching);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree();
                }
            }
        }
        offspring_size_pvt = variator_limit_pvt;

        // RDO
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->rdo_proportion), (size_t) population.size());
#pragma omp parallel for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeRDO(*selected_parents[i], conf->maximum_tree_height, fitness->TrainX, fitness->TrainY, *semlib, conf->uniform_depth_variation, conf->caching, conf->linear_scaling);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree();
                }
            }
        }
        offspring_size_pvt = variator_limit_pvt;

        // AGX
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->agx_proportion), (size_t) population.size());
#pragma omp parallel for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i += 2) {
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeAGX(*selected_parents[i], *selected_parents[randu() * selected_parents.size()], conf->maximum_tree_height, fitness->TrainX, *semlib, conf->uniform_depth_variation, conf->caching, conf->linear_scaling);
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = selected_parents[i]->CloneSubtree();
            }
            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree();
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree();
                }
            } else
                oo.second->ClearSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

        // Reproduction
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->reproduction_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            offspring[i] = selected_parents[i]->CloneSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

#pragma omp single 
        {
            offspring_size = offspring_size_pvt;
            variator_limit = variator_limit_pvt;
        }
    }

    // elitism
    // size_t actual_elitism = max(conf->elitism, population.size() - offspring_size); // e.g., when crossover delivers 1 child less
    // if (actual_elitism > 0) {
    //     vector<size_t> elitism_indices;
    //     elitism_indices.reserve(actual_elitism);
    //     // Add indices of not initialized offspring
    //     for (size_t i = variator_limit; i < population.size(); i++)
    //         elitism_indices.push_back(i);
    //     // If more elites need to be used, random members of the offspring will be replaced
    //     if (elitism_indices.size() < actual_elitism)
    //         for (size_t i = 0; i < actual_elitism; i++)
    //             elitism_indices.push_back(randu() * population.size());
    //     // sort fitnesses
    //     vec fitnesses = fitness->GetPopulationFitness(population, false, conf->caching);
    //     uvec order_fitnesses = sort_index(fitnesses);
    //     // insert elites
    //     size_t j = 0;
    //     for (size_t i : elitism_indices) {
    //         if (offspring[i])
    //             offspring[i]->ClearSubtree();
    //         offspring[i] = population[ order_fitnesses[j++] ]->CloneSubtree();
    //     }
    // }
    std::cout<<"in makeoffspring function, the final offspring vector size is:"<<offspring.size()<<std::endl;
    assert(offspring.size() == (population.size()-num_elites));

    return offspring;
}






void GenerationHandler::MakeOffspring(const std::vector<Node *> & population, const vector<Node*> & selected_parents, vector<Node*>& offspring) {

    std::cout<<"    go into MakeOffspring function"<<std::endl;
    offspring.clear();
    offspring.shrink_to_fit();
    offspring.assign(population.size(),nullptr);
    //std::vector<Node *> offspring(population.size(), NULL);

    // Variation
    size_t offspring_size, variator_limit;
#pragma omp parallel
    {
        size_t offspring_size_pvt = 0;
        size_t variator_limit_pvt = population.size() * conf->subtree_crossover_proportion;
        size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

        max_attempts_variation = 10;
        attempts_variation = 0;

        population_chunk = (variator_limit_pvt / omp_get_num_threads());
        population_chunk_start_idx = population_chunk * omp_get_thread_num();

        // crossover
#pragma omp for schedule(static)
        for (size_t i = 0; i < variator_limit_pvt; i += 2) {
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeCrossover(*selected_parents[i], *selected_parents[population_chunk_start_idx + randu() * population_chunk], conf->uniform_depth_variation, conf->caching);
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = selected_parents[i]->CloneSubtree();
            }
            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree();
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree();
                }
            } else
                oo.second->ClearSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

        // mutation
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->subtree_mutation_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeMutation(*selected_parents[i], *tree_initializer, conf->functions, conf->terminals, conf->initial_maximum_tree_height, conf->uniform_depth_variation, conf->caching);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree();
                }
            }
        }
        offspring_size_pvt = variator_limit_pvt;

        // RDO
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->rdo_proportion), (size_t) population.size());
#pragma omp parallel for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeRDO(*selected_parents[i], conf->maximum_tree_height, fitness->TrainX, fitness->TrainY, *semlib, conf->uniform_depth_variation, conf->caching, conf->linear_scaling);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree();
                }
            }
        }
        offspring_size_pvt = variator_limit_pvt;

        // AGX
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->agx_proportion), (size_t) population.size());
#pragma omp parallel for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i += 2) {
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeAGX(*selected_parents[i], *selected_parents[randu() * selected_parents.size()], conf->maximum_tree_height, fitness->TrainX, *semlib, conf->uniform_depth_variation, conf->caching, conf->linear_scaling);
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = selected_parents[i]->CloneSubtree();
            }
            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree();
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree();
                }
            } else
                oo.second->ClearSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

        // Reproduction
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->reproduction_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            offspring[i] = selected_parents[i]->CloneSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

#pragma omp single 
        {
            offspring_size = offspring_size_pvt;
            variator_limit = variator_limit_pvt;
        }
    }

    // elitism
    size_t actual_elitism = max(conf->elitism, population.size() - offspring_size); // e.g., when crossover delivers 1 child less

    if (actual_elitism > 0) {
        vector<size_t> elitism_indices;
        elitism_indices.reserve(actual_elitism);
        // Add indices of not initialized offspring
        for (size_t i = variator_limit; i < population.size(); i++)
            elitism_indices.push_back(i);
        // If more elites need to be used, random members of the offspring will be replaced
        if (elitism_indices.size() < actual_elitism)
            for (size_t i = 0; i < actual_elitism; i++)
                elitism_indices.push_back(randu() * population.size());
        // sort fitnesses
        vec fitnesses = fitness->GetPopulationFitness(population, false, conf->caching);
        uvec order_fitnesses = sort_index(fitnesses);
        // insert elites
        size_t j = 0;
        for (size_t i : elitism_indices) 
        {
            if (offspring[i])
                offspring[i]->ClearSubtree();
            offspring[i] = population[ order_fitnesses[j++] ]->CloneSubtree();
        }
    }

    assert(offspring.size() == population.size());
    std::cout<<"    return from MakeOffspring function"<<std::endl;
    //return offspring;
    return;
}




std::vector<Node*> GenerationHandler::MakeOffspring(const std::vector<Node *> & population, const std::vector<Node*> & selected_parents) {

    int num_clear = 0;
    int num_clone = 0;
    int num_create = 0;
    
    int num_crossover = 0;
    int num_mutation = 0;
    int num_other = 0;

    std::cout<<"    go into MakeOffspring function"<<std::endl;
    std::vector<Node *> offspring(population.size(), NULL);

    // Variation
    size_t offspring_size, variator_limit;
#pragma omp parallel
    {
        size_t offspring_size_pvt = 0;
        size_t variator_limit_pvt = population.size() * conf->subtree_crossover_proportion;
        size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

        max_attempts_variation = 10;
        attempts_variation = 0;

        population_chunk = (variator_limit_pvt / omp_get_num_threads());
        population_chunk_start_idx = population_chunk * omp_get_thread_num();

        // crossover
#pragma omp for schedule(static)
        for (size_t i = 0; i < variator_limit_pvt; i += 2) {
            num_crossover++;
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeCrossover(*selected_parents[i], *selected_parents[population_chunk_start_idx + randu() * population_chunk], conf->uniform_depth_variation, conf->caching);
            num_create += 2;
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree(false);num_clear++;
                offspring[i] = selected_parents[i]->CloneSubtree(false);num_clone++;
            }
            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree(false);num_clear++;
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree(false);num_clone++;
                }
            } else{
                oo.second->ClearSubtree(false);num_clear++;
                num_create--;
            }
        }
        offspring_size_pvt = variator_limit_pvt;
        // mutation
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->subtree_mutation_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            num_mutation++;
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeMutation(*selected_parents[i], *tree_initializer, conf->functions, conf->terminals, conf->initial_maximum_tree_height, conf->uniform_depth_variation, conf->caching);
                num_create++;
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree(false);num_clear++;
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation){
                        offspring[i] = selected_parents[i]->CloneSubtree(false);num_clone++;
                    }
                }
            }
        }
        offspring_size_pvt = variator_limit_pvt;
        // RDO
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->rdo_proportion), (size_t) population.size());
#pragma omp parallel for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            num_other++;
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeRDO(*selected_parents[i], conf->maximum_tree_height, fitness->TrainX, fitness->TrainY, *semlib, conf->uniform_depth_variation, conf->caching, conf->linear_scaling);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree(true);
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree(true);
                }
            }
        }
        offspring_size_pvt = variator_limit_pvt;
        // AGX
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->agx_proportion), (size_t) population.size());
#pragma omp parallel for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i += 2) {
            num_other++;
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeAGX(*selected_parents[i], *selected_parents[randu() * selected_parents.size()], conf->maximum_tree_height, fitness->TrainX, *semlib, conf->uniform_depth_variation, conf->caching, conf->linear_scaling);
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree(true);
                offspring[i] = selected_parents[i]->CloneSubtree(true);
            }
            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree(true);
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree(true);
                }
            } else
                oo.second->ClearSubtree();
        }
        offspring_size_pvt = variator_limit_pvt;

        // Reproduction
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->reproduction_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            num_other++;
            offspring[i] = selected_parents[i]->CloneSubtree(true);
        }
        offspring_size_pvt = variator_limit_pvt;

#pragma omp single 
        {
            offspring_size = offspring_size_pvt;
            variator_limit = variator_limit_pvt;
        }
    }
    // elitism
    size_t actual_elitism = max(conf->elitism, population.size() - offspring_size); // e.g., when crossover delivers 1 child less

    if (actual_elitism > 0) {
        std::cout<<"elitism in makeoffspring is used"<<std::endl;
        vector<size_t> elitism_indices;
        elitism_indices.reserve(actual_elitism);
        // Add indices of not initialized offspring
        for (size_t i = variator_limit; i < population.size(); i++)
            elitism_indices.push_back(i);
        // If more elites need to be used, random members of the offspring will be replaced
        if (elitism_indices.size() < actual_elitism)
            for (size_t i = 0; i < actual_elitism; i++)
                elitism_indices.push_back(randu() * population.size());
        // sort fitnesses
        vec fitnesses = fitness->GetPopulationFitness(population, false, conf->caching);
        uvec order_fitnesses = sort_index(fitnesses);
        // insert elites
        size_t j = 0;
        for (size_t i : elitism_indices) 
        {
            if (offspring[i])
                offspring[i]->ClearSubtree(true);
            offspring[i] = population[ order_fitnesses[j++] ]->CloneSubtree(true);
        }
    }

    std::cout<<"num_crossover="<<num_crossover<<", num_mutation="<<num_mutation<<", num_other="<<num_other<<std::endl;
    std::cout<<"num_clear="<<num_clear<<", num_clone="<<num_clone<<", num_create="<<num_create<<std::endl;
    assert(offspring.size() == population.size());
    std::cout<<"    return from MakeOffspring function"<<std::endl;
    return offspring;
}

bool GenerationHandler::ValidateOffspring(Node* offspring, int max_height, int  max_size) {

    if (max_height > -1) {
        if (offspring->GetHeight() > max_height) {
            return false;
        }
    }

    if (max_size > -1) {
        size_t size = offspring->GetSubtreeNodes(true).size();
        if (size > max_size) {
            return false;
        }
    }

    return true;
}

bool GenerationHandler::CheckPopulationConverged(const std::vector<Node*>& population) {
    return false;
}

std::vector<Node*> GenerationHandler::MakeOffspring_counting_improvment_frequency_with_ExpLenRestriction(const std::vector<Node *> & population,  const std::vector<Node*> & candidate1_list_crossover, const std::vector<Node *> candidate2_list_crossover, const std::vector<Node*> & candidate_list_mutation,  std::vector<double> mutation_ExpLength_FitnessAttempt_frequency, std::vector<std::vector<double>> crossover_ExpLength_FitnessAttempt_frequency)
{
    std::vector<Node *> offspring(population.size(), NULL);
    size_t offspring_size, variator_limit;
#pragma omp parallel
    {
        size_t offspring_size_pvt = 0;
        size_t variator_limit_pvt = population.size() * conf->subtree_crossover_proportion;
        size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

        max_attempts_variation = 10;
        attempts_variation = 0;

        population_chunk = (variator_limit_pvt / omp_get_num_threads());
        population_chunk_start_idx = population_chunk * omp_get_thread_num();
        // crossover
        int cursor = 0;
#pragma omp for schedule(static)
        for (size_t i = 0; i < variator_limit_pvt; i += 2) {
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeCrossover(*candidate1_list_crossover[cursor], *candidate2_list_crossover[cursor], conf->uniform_depth_variation, conf->caching);
            
            //---calculate the frequency that offspring can improve parents' quality
            int parent1_len = candidate1_list_crossover[cursor]->GetSubtreeNodes(true).size();
            int parent2_len = candidate2_list_crossover[cursor]->GetSubtreeNodes(true).size();
            crossover_ExpLength_FitnessAttempt_frequency[parent1_len-1][parent2_len-1] ++;
            
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = candidate1_list_crossover[cursor]->CloneSubtree();
            }
            offspring[i]->parent_exp_len = parent1_len;
            offspring[i]->parent_fitness =  candidate1_list_crossover[cursor]->cached_fitness;
            offspring[i]->parent2_exp_len = parent2_len;
            offspring[i]->parent2_fitness =  candidate2_list_crossover[cursor]->cached_fitness;

            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree();
                    offspring[i + 1] = candidate2_list_crossover[cursor]->CloneSubtree();
                }
                //attention: the most part of gene of the second offspring comes from the second parent, this way we should swap the location for storing parent1 and parent2 here
                offspring[i + 1]->parent_exp_len = parent2_len;
                offspring[i + 1]->parent_fitness =  candidate1_list_crossover[cursor]->cached_fitness;
                offspring[i + 1]->parent2_exp_len = parent1_len;
                offspring[i + 1]->parent2_fitness =  candidate2_list_crossover[cursor]->cached_fitness;
                crossover_ExpLength_FitnessAttempt_frequency[parent2_len-1][parent1_len-1] ++;
            } else
                oo.second->ClearSubtree();
            cursor++;
        }
        offspring_size_pvt = variator_limit_pvt;

        // mutation
        cursor = 0;
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->subtree_mutation_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeMutation(*candidate_list_mutation[cursor], *tree_initializer, conf->functions, conf->terminals, conf->initial_maximum_tree_height, conf->uniform_depth_variation, conf->caching);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = candidate_list_mutation[cursor]->CloneSubtree();
                }
                
            }
            int parent_len = candidate_list_mutation[cursor]->GetSubtreeNodes(true).size();
            offspring[i]->parent_fitness = candidate_list_mutation[cursor]->cached_fitness;
            offspring[i]->parent_exp_len = parent_len;
            offspring[i]->parent2_fitness = -1;
            offspring[i]->parent2_exp_len = -1;
            mutation_ExpLength_FitnessAttempt_frequency[parent_len-1] ++;
            cursor++;
        }
        offspring_size_pvt = variator_limit_pvt;
    }
    return offspring;
}




std::vector<Node*> GenerationHandler::MakeOffspring_counting_improvment_frequency(const std::vector<Node *> & population, const std::vector<Node*> & selected_parents,  std::vector<double> & mutation_ExpLength_FitnessAttempt_frequency, std::vector<std::vector<double>> & crossover_ExpLength_FitnessAttempt_frequency)
{
    //std::cout<<"MakeOffspring_counting_improvment_frequency flag1"<<std::endl;
    std::vector<Node *> offspring(population.size(), NULL);
    // Variation
    size_t offspring_size, variator_limit;
#pragma omp parallel
    {
        size_t offspring_size_pvt = 0;
        size_t variator_limit_pvt = population.size() * conf->subtree_crossover_proportion;
        size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

        max_attempts_variation = 10;
        attempts_variation = 0;

        population_chunk = (variator_limit_pvt / omp_get_num_threads());
        population_chunk_start_idx = population_chunk * omp_get_thread_num();
        //std::cout<<"MakeOffspring_counting_improvment_frequency flag2"<<std::endl;
        // crossover
#pragma omp for schedule(static)
        for (size_t i = 0; i < variator_limit_pvt; i += 2) {
            int idx = population_chunk_start_idx + randu() * population_chunk;
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeCrossover(*selected_parents[i], *selected_parents[idx], conf->uniform_depth_variation, conf->caching);
            
            //---calculate the frequency that offspring can improve parents' quality
            int parent1_len = selected_parents[i]->GetSubtreeNodes(true).size();
            int parent2_len = selected_parents[idx]->GetSubtreeNodes(true).size();
            crossover_ExpLength_FitnessAttempt_frequency[parent1_len-1][parent2_len-1] ++;
            
            offspring[i] = oo.first;
            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = selected_parents[i]->CloneSubtree();
            }
            offspring[i]->parent_exp_len = parent1_len;
            offspring[i]->parent_fitness =  selected_parents[i]->cached_fitness;
            offspring[i]->parent2_exp_len = parent2_len;
            offspring[i]->parent2_fitness =  selected_parents[idx]->cached_fitness;

            if (i < variator_limit_pvt) {
                offspring[i + 1] = oo.second;
                if (!ValidateOffspring(offspring[i + 1], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i + 1]->ClearSubtree();
                    offspring[i + 1] = selected_parents[i + 1]->CloneSubtree();
                }
                //attention: the most part of gene of the second offspring comes from the second parent, this way we should swap the location for storing parent1 and parent2 here
                offspring[i + 1]->parent_exp_len = parent2_len;
                offspring[i + 1]->parent_fitness =  selected_parents[idx]->cached_fitness;
                offspring[i + 1]->parent2_exp_len = parent1_len;
                offspring[i + 1]->parent2_fitness =  selected_parents[i]->cached_fitness;
                crossover_ExpLength_FitnessAttempt_frequency[parent2_len-1][parent1_len-1] ++;
            } else
                oo.second->ClearSubtree();
        }
        //std::cout<<"MakeOffspring_counting_improvment_frequency flag3"<<std::endl;
        offspring_size_pvt = variator_limit_pvt;

        // mutation
        //std::cout<<"MakeOffspring_counting_improvment_frequency flag4"<<std::endl;
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->subtree_mutation_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeMutation(*selected_parents[i], *tree_initializer, conf->functions, conf->terminals, conf->initial_maximum_tree_height, conf->uniform_depth_variation, conf->caching);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree();
                }
            }
            int parent_len = selected_parents[i]->GetSubtreeNodes(true).size();
            offspring[i]->parent_fitness = selected_parents[i]->cached_fitness;
            offspring[i]->parent_exp_len = parent_len;
            offspring[i]->parent2_fitness = -1;
            offspring[i]->parent2_exp_len = -1;
            mutation_ExpLength_FitnessAttempt_frequency[parent_len-1] ++;
        }
        //std::cout<<"MakeOffspring_counting_improvment_frequency flag5"<<std::endl;
        offspring_size_pvt = variator_limit_pvt;
    }
    std::cout<<"MakeOffspring_counting_improvment_frequency finished, offspring size:"<<offspring.size()<<std::endl;
    return offspring;
}






std::vector<Node*> GenerationHandler::MakeOffspring_counting_improvment_frequency_with_LCTrunction(const std::vector<Node *> & population, const std::vector<Node *> & selected_parents, std::vector<double> & crossover_attempts, std::vector<double> & mutation_attempts)
{

    std::cout<<"MakeOffspring_counting_improvment_frequency_with_LCTrunction..."<<std::endl;
    std::vector<Node *> offspring(population.size(), NULL);

    // Variation
    size_t offspring_size, variator_limit;
#pragma omp parallel
    {
        size_t offspring_size_pvt = 0;
        size_t variator_limit_pvt = population.size() * conf->subtree_crossover_proportion;
        size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

        max_attempts_variation = 10;
        attempts_variation = 0;

        population_chunk = (variator_limit_pvt / omp_get_num_threads());
        population_chunk_start_idx = population_chunk * omp_get_thread_num();
        // crossover
#pragma omp for schedule(static)
        for (size_t i = 0; i < variator_limit_pvt; i += 1) {
            pair<Node*, Node*> oo = SubtreeVariator::SubtreeCrossover(*selected_parents[i], *selected_parents[population_chunk_start_idx + randu() * population_chunk], conf->uniform_depth_variation, conf->caching);
            oo.second->ClearSubtree();
            offspring[i] = oo.first;
            int parent_len = selected_parents[i]->GetSubtreeNodes(true).size();
            crossover_attempts[parent_len-1] ++;

            if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                offspring[i]->ClearSubtree();
                offspring[i] = selected_parents[i]->CloneSubtree();
            }
            offspring[i]->parent_exp_len = parent_len;
            offspring[i]->parent_fitness =  selected_parents[i]->cached_fitness;   
            offspring[i]->generate_type = 0;
        }
        offspring_size_pvt = variator_limit_pvt;
        // mutation
        variator_limit_pvt = min((size_t) (variator_limit_pvt + population.size() * conf->subtree_mutation_proportion), (size_t) population.size());
#pragma omp for schedule(static)
        for (size_t i = offspring_size_pvt; i < variator_limit_pvt; i++) {
            while (offspring[i] == NULL) {
                offspring[i] = SubtreeVariator::SubtreeMutation(*selected_parents[i], *tree_initializer, conf->functions, conf->terminals, conf->initial_maximum_tree_height, conf->uniform_depth_variation, conf->caching);
                if (!ValidateOffspring(offspring[i], conf->maximum_tree_height, conf->maximum_solution_size)) {
                    offspring[i]->ClearSubtree();
                    offspring[i] = NULL;
                    attempts_variation++;
                    if (attempts_variation >= max_attempts_variation)
                        offspring[i] = selected_parents[i]->CloneSubtree();
                }
            }
            int parent_len = selected_parents[i]->GetSubtreeNodes(true).size();
            mutation_attempts[parent_len - 1] ++;
            offspring[i]->parent_fitness =  selected_parents[i]->cached_fitness; 
            offspring[i]->generate_type = 1;
            offspring[i]->parent_exp_len = double_t(parent_len);
            //std::cout<< offspring[i]->parent_exp_len<<std::endl;
        }
        offspring_size_pvt = variator_limit_pvt;
    }
    std::cout<<"finished, offspring size:"<<offspring.size();
    assert(offspring.size() == population.size());
    std::cout<<" ,return to PerformGeneration"<<std::endl;
    return offspring;
}