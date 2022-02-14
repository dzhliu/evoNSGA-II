/*
 


 */

/* 
 * File:   TournamentSelection.cpp
 * Author: virgolin
 * 
 * Created on June 28, 2018, 3:04 PM
 */

#include "GPGOMEA/Selection/TournamentSelection.h"

using namespace std;
using namespace arma;

int TournamentSelection::sampling_from_descrete_probibality_distribution(std::vector<double> probability, std::vector<std::vector<Node *>> id_exp_len_list)
{
    std::cout<<"go into sampling_from_descrete_probibality_distribution"<<std::endl;
    double sum_probability = 0;
    std::vector<double> normailzed_probability;
    for(int i = 0; i < probability.size(); i++)
    {
        sum_probability += probability[i];
        normailzed_probability.push_back(probability[i]);
    }
    
    std::cout<<"======================="<<std::endl;
    //如果某个表达式长度有概率没个体，则删除之
    for(int i = 0; i < id_exp_len_list.size(); i++)
    {
        std::cout<<"exp len_seq="<<i;
        if(id_exp_len_list[i].size() == 0)
        {
            std::cout<<" , no exp len, prob =";
            std::cout<<normailzed_probability[i];
            sum_probability -= probability[i];
            normailzed_probability[i] = 0;
        }
        std::cout<<std::endl;
    }
    std::cout<<"======================="<<std::endl;
    if(sum_probability == 0)
    {
        std::cout<<"ERROR in sampling_from_joint_descrete_probibality_distribution, sum_check = 0"<<std::endl;
        exit(-1);
    }


    for(int i = 0; i < probability.size(); i++)
    {
        normailzed_probability[i] = normailzed_probability[i]/sum_probability;
    }
    double rand_value = arma::randu();
    sum_probability = 0;
    int len = -1;
    for(int i = 0; i < normailzed_probability.size(); i++)
    {
        if(rand_value < sum_probability + normailzed_probability[i])
        {
            len = i;
            break;
        }
        else
        {
            sum_probability += normailzed_probability[i];
        }
    }

    if(len == -1)
    {
        std::cout<<"error in sampling_from_descrete_probibality_distribution, rand="<<rand<<", sum_probability="<<sum_probability<<std::endl;
        exit(-1);
    }
    return len;
}

std::pair<int,int> TournamentSelection::sampling_from_joint_descrete_probibality_distribution(std::vector<vector<double>> probability, std::vector<std::vector<Node *>> id_exp_len_list)
{
    std::cout<<"go into sampling_from_joint_descrete_probibality_distribution"<<std::endl;



    std::cout<<"show number of individuals with each expression length"<<std::endl;
    for(int i = 0; i < id_exp_len_list.size(); i++)
    {
        std::cout<<id_exp_len_list[i].size()<<" ";
    }
    std::cout<<std::endl;


    double sum_check_row = 0;
    double sum_check_col = 0;

    double sum_probibality_p1 = 0;
    double sum_probibality_p2 = 0;
    std::vector<double> normailzed_probability_p1;
    std::vector<double> normailzed_probability_p2;
    for(int i = 0; i < probability.size(); i++)
    {
        double sum_row = 0;
        double sum_col = 0;
        for(int j = 0; j < probability.size(); j++)
        {
            sum_row += probability[i][j];
            sum_col += probability[j][i];
        }
        sum_probibality_p1 += sum_row;
        sum_probibality_p2 += sum_col;
        normailzed_probability_p1.push_back(sum_row);
        normailzed_probability_p2.push_back(sum_col);
        sum_check_row+= sum_row;
        sum_check_col+= sum_col;
    }

    std::cout<<"======================="<<std::endl;
    //如果某个表达式长度有概率没个体，则删除之
    for(int i = 0; i < id_exp_len_list.size(); i++)
    {
        if(id_exp_len_list[i].size() == 0)
        {
            std::cout<<"exp len seq="<<i<<", no exp len, sum_prob for p1="<<normailzed_probability_p1[i]<<", sum_prob for p2="<<normailzed_probability_p2[i]<<std::endl;
            sum_probibality_p1 = sum_probibality_p1 - normailzed_probability_p1[i];
            sum_probibality_p2 = sum_probibality_p2 - normailzed_probability_p2[i];

            normailzed_probability_p1[i] = 0;
            normailzed_probability_p2[i] = 0;
        }
    }
    std::cout<<"======================="<<std::endl;
    if(sum_check_row == 0 || sum_check_col == 0)
    {
        std::cout<<"ERROR in sampling_from_joint_descrete_probibality_distribution, sum_check = 0"<<std::endl;
        exit(-1);
    }

    std::cout<<"before normailze, probability_p1:"<<std::endl;
    for(int i = 0; i < normailzed_probability_p1.size(); i++)
        std::cout<<normailzed_probability_p1[i]<<" ";
    std::cout<<endl;
    std::cout<<"before normailze, probability_p2:"<<std::endl;
    for(int i = 0; i < normailzed_probability_p2.size(); i++)
        std::cout<<normailzed_probability_p2[i]<<" ";
    std::cout<<endl;


    //even if there is record for some expression length, yet they may be disappera now, so the marginalized probability vector should be filtered again to delete any probibality value which there is no individuals in the population match with




    for(int i = 0; i < normailzed_probability_p1.size(); i++)
    {
        if(sum_probibality_p1 != 0)
            normailzed_probability_p1[i] = normailzed_probability_p1[i]/sum_probibality_p1;
        if(sum_probibality_p2 != 0)
            normailzed_probability_p2[i] = normailzed_probability_p2[i]/sum_probibality_p2;
    }

    std::cout<<"normailzed_probability_p1:"<<std::endl;
    for(int i = 0; i < normailzed_probability_p1.size(); i++)
        std::cout<<normailzed_probability_p1[i]<<" ";
    std::cout<<endl;
    std::cout<<"normailzed_probability_p2:"<<std::endl;
    for(int i = 0; i < normailzed_probability_p2.size(); i++)
        std::cout<<normailzed_probability_p2[i]<<" ";
    std::cout<<endl;
    
    
    
    std::cout<<"the generated probibality matrix for mutation"<<std::endl;
    for(int i = 0; i < normailzed_probability_p1.size(); i++)
        std::cout<<normailzed_probability_p1[i]<<" ";
    std::cout<<std::endl;

    std::cout<<"the generated probibality matrix for crossover"<<std::endl;
    for(int i = 0; i < probability.size(); i++)
    {
        for(int j = 0; j < probability.size(); j++)
        {
            std::cout<<probability[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    
    std::cout<<"sampling_from_joint_descrete_probibality_distribution flag1"<<std::endl;
    double rand1 = arma::randu();
    std::cout<<"rand1="<<rand1<<std::endl;
    while(rand1 >= 1 || rand1 <= 0)
    {
        std::cout<<"rand1="<<rand1<<", regenerate:";
        rand1 = arma::randu();
        std::cout<<rand1<<std::endl;
    }
    double sum_probibality1 = 0;
    int len_exp1 = -1;
    std::cout<<"sampling_from_joint_descrete_probibality_distribution flag2"<<std::endl;

    for(int i = 0; i < normailzed_probability_p1.size(); i++)
    {
        if(rand1 < sum_probibality1 + normailzed_probability_p1[i])
        {
            len_exp1 = i;
            break;
        }
        else
        {
            sum_probibality1 += normailzed_probability_p1[i];
        }
    }
    std::cout<<"sampling_from_joint_descrete_probibality_distribution flag3"<<std::endl;
    double rand2 = arma::randu();
    std::cout<<"rand2="<<rand2<<std::endl;
    while(rand2 >= 1 || rand2 <= 0){
        std::cout<<"rand2="<<rand2<<", regenerate:";
        rand2 = arma::randu();
        std::cout<<rand2<<std::endl;
    }
    std::cout<<"sampling_from_joint_descrete_probibality_distribution flag4"<<std::endl;
    double sum_probibality2 = 0;
    int len_exp2 = -1;
    for(int i = 0; i < normailzed_probability_p2.size(); i++)
    {
        if(rand2 <= sum_probibality2 + normailzed_probability_p2[i])
        {
            len_exp2 = i;
            break;
        }
        else
        {
            sum_probibality2 += normailzed_probability_p2[i];
        }
    }
    std::cout<<"sampling_from_joint_descrete_probibality_distribution flag5"<<std::endl;
    std::pair<int,int> exp_len;
    exp_len.first = len_exp1;
    exp_len.second = len_exp2;
    if(len_exp1 < 0 || len_exp2 < 0)
    {
        std::cout<<"error in sampling_from_joint_descrete_probibality_distribution, rand1 num= "<<rand1<<", rand2="<<rand2<<" ,len_exp1="<<len_exp1<<", len_exp2="<<len_exp2<<", sum_probibality1="<<sum_probibality1<<", sum_probibality2"<<sum_probibality2<<std::endl;
        exit(-1);
    }
    
    double row_sum_value = 0;
    double col_sum_value = 0;
    for(int i = 0; i < probability.size(); i++)
    {
        row_sum_value+= probability[len_exp1][i];
        col_sum_value+= probability[i][len_exp2];
    }
    if(row_sum_value == 0 || col_sum_value == 0)
    {
        std::cout<<"error, an empty row or col of probibality matrix has been selected, row="<<len_exp1<<", col="<<len_exp2<<std::endl;
        std::cout<<"row_sum_value= "<<row_sum_value<<", col_sum_value="<<col_sum_value<<std::endl;
        exit(-1);
    }

    std::cout<<"in sampling_from_joint_descrete_probibality_distribution, rand1 num= "<<rand1<<", rand2="<<rand2<<" ,len_exp1="<<len_exp1<<", len_exp2="<<len_exp2<<", sum_probibality1="<<sum_probibality1<<", sum_probibality2="<<sum_probibality2<<std::endl;
    std::cout<<"sampling_from_joint_descrete_probibality_distribution finished"<<std::endl;
    return exp_len;
}


Node * TournamentSelection::GetTournamentSelectionWinner(const std::vector<Node*>& candidates, size_t tournament_size) {
    Node * winner = candidates[ arma::randu() * candidates.size() ];
    double_t winner_fit = winner->cached_fitness;

    for (size_t i = 1; i < tournament_size; i++) {
        Node * candidate = candidates[ arma::randu() * candidates.size() ];
        if (candidate->cached_fitness < winner_fit) {
            winner = candidate;
            winner_fit = candidate->cached_fitness;
        }
    }
    
    return winner;
}

//revised version to select individuals based on both fitness and also the frequency of fitness improvment
void TournamentSelection::GetMOExpLengthSelectionWinner_BasedOnJointProbibality(const std::vector<Node*>& candidates,int tournament_size, int max_exp_length, std::vector<double> frequency_matrix_mutation, std::vector<std::vector<double>> frequency_matrix_crossover,  std::vector<Node *> & candidate1_list_crossover,  std::vector<Node *> & candidate2_list_crossover, std::vector<Node *> & candidate_list_mutation) 
{
    std::cout<<"GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag1"<<std::endl;
    //two lists for crossover and another one list for mutation
    //std::vector<Node *> candidate1_list_crossover;
    //std::vector<Node *> candidate2_list_crossover;
    //std::vector<Node *> candidate_list_mutation;
    
    //divide the quantity of population for mutation and crossover
    double crossover_proportion = 0.5;
    double mutation_proportion = 0.5;
    int num_crossover = candidates.size()*crossover_proportion*0.5;
    int num_mutation = candidates.size()*mutation_proportion;

    if(candidates.size()%2 != 0)
    {
        num_crossover++;
    }
    
    //check if the quantity of population for mutation and crossover is correct
    if((num_crossover*2 + num_mutation)!=candidates.size())
    {
        std::cout<<"calculate number of individuals should be generated for crossover/mutation is wrong"<<std::endl;
        exit(-1);
    }

    std::cout<<"GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag2"<<std::endl;
    //建立种群中各个个体的表达式长度与个体在种群中序号的映射表
    std::vector<std::vector<Node *>> id_exp_len_list; //id_exp_len_list[i] contains id of individuals in candidates (the population) with expression length i+1  !!!! length is i+1 becuase the index start from 0
    for(int i = 0; i < max_exp_length; i++)
    {
        vector<Node *> vec;
        id_exp_len_list.push_back(vec);
    }
    for(int i = 0; i < candidates.size(); i++)
    {
        int exp_len = candidates[i]->GetSubtreeNodes(true).size();
        id_exp_len_list[exp_len-1].push_back(candidates[i]);
    }

    std::cout<<"show number of individuals with each expression length"<<std::endl;
    for(int i = 0; i < id_exp_len_list.size(); i++)
    {
        std::cout<<id_exp_len_list[i].size()<<" ";
    }
    std::cout<<std::endl;


    std::cout<<"GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3"<<std::endl;
    //select individuals for crossover
    for(int i = 0; i < num_crossover; i++)
    {
        std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1"<<std::endl;
        pair<int,int> parents_len = sampling_from_joint_descrete_probibality_distribution(frequency_matrix_crossover, id_exp_len_list);
        std::cout<<"        --->len1:"<<parents_len.first<<",--->len2:"<<parents_len.second<<std::endl;
        if(id_exp_len_list[parents_len.first].size() == 0 || id_exp_len_list[parents_len.second].size() == 0)
        {
            std::cout<<"error"<<std::endl;
            std::cout<<"parents1_len="<<parents_len.first<<", parents2_len="<<parents_len.second<<std::endl;
            std::cout<<id_exp_len_list[parents_len.first].size()<<","<<id_exp_len_list[parents_len.second].size()<<std::endl;
            exit(-1);
        }
        //如果两个个体选自同一表达式长度集合，而集合中仅仅有一个个体，这时候会重复删除个体两次，删除第二次的时候数组越界
        if(parents_len.first == parents_len.second && id_exp_len_list[parents_len.first].size() == 1)
        {
            
        }
        std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b"<<std::endl;
        if(id_exp_len_list[parents_len.first].size() < tournament_size){
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b-1"<<std::endl;
            std::pair<Node *, int> res = GetMOTournamentSelectionWinner_with_idx(id_exp_len_list[parents_len.first],id_exp_len_list[parents_len.first].size());
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b-2"<<std::endl;
            candidate1_list_crossover.push_back(res.first);
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b-3"<<std::endl;
            //id_exp_len_list[parents_len.first].erase(id_exp_len_list[parents_len.first].begin()+res.second);
        }
        else{
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b2-1"<<std::endl;
            std::pair<Node *, int> res = GetMOTournamentSelectionWinner_with_idx(id_exp_len_list[parents_len.first],tournament_size);
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b2-2"<<std::endl;
            candidate1_list_crossover.push_back(res.first);
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b2-3"<<std::endl;
            //id_exp_len_list[parents_len.first].erase(id_exp_len_list[parents_len.first].begin()+res.second);
            std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-1b2-4"<<std::endl;
        }
        std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-2";
        if(id_exp_len_list[parents_len.second].size() < tournament_size){
            std::cout<<"-branch1"<<std::endl;
            std::pair<Node *, int> res = GetMOTournamentSelectionWinner_with_idx(id_exp_len_list[parents_len.second],id_exp_len_list[parents_len.second].size());
            candidate2_list_crossover.push_back(res.first);
            std::cout<<"id_exp_len_list[i] size="<<id_exp_len_list[parents_len.second].size()<<std::endl;
            std::cout<<"res.second="<<res.second<<std::endl;
            //id_exp_len_list[parents_len.second].erase(id_exp_len_list[parents_len.second].begin()+res.second);
        }
        else{
            std::cout<<"-branch2"<<std::endl;
            std::pair<Node *, int> res = GetMOTournamentSelectionWinner_with_idx(id_exp_len_list[parents_len.second],tournament_size);
            candidate2_list_crossover.push_back(res.first);
            //id_exp_len_list[parents_len.second].erase(id_exp_len_list[parents_len.second].begin()+res.second);
        }
        std::cout<<"    GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag3-3"<<std::endl;
    }

    std::cout<<"GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag4"<<std::endl;
    //建立种群中各个个体的表达式长度与个体在种群中序号的映射表
    id_exp_len_list.clear(); //id_exp_len_list[i] contains id of individuals in candidates (the population) with expression length i+1  !!!! length is i+1 becuase the index start from 0
    for(int i = 0; i < max_exp_length; i++)
    {
        vector<Node *> vec;
        id_exp_len_list.push_back(vec);
    }
    for(int i = 0; i < candidates.size(); i++)
    {
        int exp_len = candidates[i]->GetSubtreeNodes(true).size();
        id_exp_len_list[exp_len-1].push_back(candidates[i]);
    }

    std::cout<<"GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag5"<<std::endl;
    //select individuals for mutation
    for(int i = 0; i < num_mutation; i++)
    {
        int exp_len;
        exp_len = sampling_from_descrete_probibality_distribution(frequency_matrix_mutation, id_exp_len_list);
         if(id_exp_len_list[exp_len].size() == 0)
        {
            std::cout<<"error"<<std::endl;
            std::cout<<"parents_len="<<exp_len<<std::endl;
            std::cout<<id_exp_len_list[exp_len].size()<<std::endl;
            exit(-1);
        }
        
        if(id_exp_len_list[exp_len].size() < tournament_size){
            std::pair<Node *, int> res = GetMOTournamentSelectionWinner_with_idx(id_exp_len_list[exp_len],id_exp_len_list[exp_len].size());
            candidate_list_mutation.push_back(res.first);
            id_exp_len_list[exp_len].erase(id_exp_len_list[exp_len].begin()+res.second);
        }
        else{
            std::pair<Node *, int> res = GetMOTournamentSelectionWinner_with_idx(id_exp_len_list[exp_len],tournament_size);
            candidate_list_mutation.push_back(res.first);
            id_exp_len_list[exp_len].erase(id_exp_len_list[exp_len].begin()+res.second);
        }
    }

    std::cout<<"GetMOExpLengthSelectionWinner_BasedOnJointProbibality flag6"<<std::endl;
    //check if enough number of individuals has been generated. if not, something wrong with my code:(
    if((candidate_list_mutation.size() + candidate1_list_crossover.size() + candidate2_list_crossover.size()) != candidates.size())
    {
        std::cout<<"error, the number of individuals selected is not enough!"<<std::endl;
        exit(-1);
    }
}

std::pair<Node *,int> TournamentSelection::GetMOTournamentSelectionWinner_with_idx(const std::vector<Node*>& candidates, size_t tournament_size) {
    std::cout<<"GetMOTournamentSelectionWinner_with_idx flag1"<<std::endl;
    int idx = arma::randu() * candidates.size();
    std::cout<<"candidates size="<<candidates.size()<<",GetMOTournamentSelectionWinner_with_idx flag2"<<std::endl;
    Node * winner = candidates[ idx ];
    for (size_t i = 1; i < tournament_size; i++) {
        std::cout<<"GetMOTournamentSelectionWinner_with_idx flag2-loop"<<std::endl;
        int idx2 = arma::randu() * candidates.size();
        Node * candidate = candidates[ idx2 ];
        if ((candidate->rank < winner->rank) || 
            (candidate->rank == winner->rank && candidate->crowding_distance > winner->crowding_distance) ) {
            winner = candidate;
            idx = idx2;
        }
    }
    std::cout<<"GetMOTournamentSelectionWinner_with_idx flag3"<<std::endl;
    std::pair<Node *, int> p;
    p.first = winner;
    p.second = idx;
    std::cout<<"GetMOTournamentSelectionWinner_with_idx finished"<<std::endl;
    return p;
}

Node * TournamentSelection::GetMOTournamentSelectionWinner(const std::vector<Node*>& candidates, size_t tournament_size) {
    Node * winner = candidates[ arma::randu() * candidates.size() ];

    for (size_t i = 1; i < tournament_size; i++) {
        Node * candidate = candidates[ arma::randu() * candidates.size() ];
        if ((candidate->rank < winner->rank) || 
            (candidate->rank == winner->rank && candidate->crowding_distance > winner->crowding_distance) ) {
            winner = candidate;
        }
    }
    
    return winner;
}

vector<Node*> TournamentSelection::PopulationWiseTournamentSelection(const std::vector<Node*> population, size_t selection_size, size_t tournament_size) {
    size_t n_pop = population.size();
    vector<Node*> selected;
    selected.reserve(selection_size);

    // make sure that n_pop is multiple of tournament_size
    assert(  ((double)n_pop) / tournament_size == (double) n_pop / tournament_size );

    size_t n_selected_per_round = n_pop / tournament_size;
    size_t n_rounds = selection_size / n_selected_per_round;

    for(size_t i = 0; i < n_rounds; i++){
        // get a random permutation 
        auto perm = arma::randperm(n_pop);

        // apply tournaments
        for(size_t j = 0; j < n_selected_per_round; j++) {
            // one tournament instance
            Node * winner = population[perm[j*tournament_size]];
            for(size_t k=j*tournament_size + 1; k < (j+1)*tournament_size; k++){
                if (population[perm[k]]->cached_fitness < winner->cached_fitness) {
                    winner = population[perm[k]];
                }
            }
            selected.push_back(winner);
        }
    }
    return selected;
}