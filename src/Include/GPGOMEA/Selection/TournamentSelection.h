/*
 


 */

/* 
 * File:   TournamentSelection.h
 * Author: virgolin
 *
 * Created on June 28, 2018, 3:04 PM
 */

#ifndef TOURNAMENTSELECTION_H
#define TOURNAMENTSELECTION_H

#include "GPGOMEA/Genotype/Node.h"

#include <armadillo>

class TournamentSelection {
public:

    static Node * GetTournamentSelectionWinner(const std::vector<Node*> & candidates, size_t tournament_size);
    static Node * GetMOTournamentSelectionWinner(const std::vector<Node*> & candidates, size_t tournament_size);
    static std::vector<Node*> PopulationWiseTournamentSelection(const std::vector<Node*> input_population, size_t selection_size, size_t tournament_size);

    static int sampling_from_descrete_probibality_distribution(std::vector<double> probability, std::vector<std::vector<Node *>> id_exp_len_list);
    static std::pair<int,int> sampling_from_joint_descrete_probibality_distribution(std::vector<std::vector<double>> probability, std::vector<std::vector<Node *>> id_exp_len_list);
    static void GetMOExpLengthSelectionWinner_BasedOnJointProbibality(const std::vector<Node*>& candidates,int tournament_size, int max_exp_length, std::vector<double> frequency_matrix_mutation, std::vector<std::vector<double>> frequency_matrix_crossover,  std::vector<Node *> & candidate1_list_crossover, std::vector<Node *> & candidate2_list_crossover, std::vector<Node *> & candidate_list_mutation);
    static std::pair<Node *,int> GetMOTournamentSelectionWinner_with_idx(const std::vector<Node*>& candidates, size_t tournament_size);


    


private:
    TournamentSelection() {};

};

#endif /* TOURNAMENTSELECTION_H */

