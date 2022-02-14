
#include "GPGOMEA/Evolution/SPEA2GenerationHandler.h"

#include <malloc.h>

using namespace std;
using namespace arma;


bool compare_pair_int_double(const std::pair<int, double> & p1, const std::pair<int, double> & p2)
{
    if(p1.second < p2.second)
        return true;
    else
        return false;
}

double SPEA2GenerationHandler::calculate_distance(Node*  n1, Node* n2)
{
    double dist = 0;
    for(int i = 0; i < n1->cached_objectives.size(); i++)
    {
        double val1 = n1->cached_objectives[i];
        double val2 = n2->cached_objectives[i];
        dist += (val2-val1)*(val2-val1);
    }
    dist = std::sqrt(dist);
    return dist;
}


void SPEA2GenerationHandler::GetStrength(std::vector<Node *> & population)
{
    // fitness->GetPopulationFitness MUST has been called before calling this function!

    
    std::vector<Node *> combined_population_archive;
    combined_population_archive.insert(combined_population_archive.end(),archive.begin(),archive.end());
    combined_population_archive.insert(combined_population_archive.end(),population.begin(),population.end());
    
    std::vector<std::vector<int>>dominance_list; //dominance_list[i] = j,k,l,m,n,... indicate the no. of individuals that dominate individual-i is j,k,l,m,n,... it worth noting that here all the no. is counted based on combined_population_archive
    for(int i = 0; i < combined_population_archive.size(); i++)
    {
        std::vector<int> vec;
        dominance_list.push_back(vec);
    }
    
    std::vector<int> strength_list(combined_population_archive.size(), 0);
    for(int i = 0; i < combined_population_archive.size(); i++)
    {
        
        Node * p = combined_population_archive[i];
        for(int j = 0; j < combined_population_archive.size(); j++)
        {
            if(i == j)
                continue;
            Node * q = combined_population_archive[j];
            if(p->Dominates(q))
                strength_list[i] = strength_list[i] + 1;
            else if (q->Dominates(p))
                dominance_list[i].push_back(j);
        }
    }
    for(int i = 0; i < combined_population_archive.size(); i++)
    {
        combined_population_archive[i]->strength = strength_list[i];           
    }
    for(int i = 0; i < combined_population_archive.size(); i++)
    {
        double sum_dominating_individuals_strength = 0;
        for(int j = 0; j < dominance_list[i].size(); j++)
        {
            sum_dominating_individuals_strength += combined_population_archive[dominance_list[i][j]]->strength;
        }
        combined_population_archive[i]->SPEA2_fitness = sum_dominating_individuals_strength;
    }
    int k = int(std::sqrt(combined_population_archive.size()));
    for(int i = 0; i < combined_population_archive.size(); i++)
    {
        std::vector<double> distance_list;
        for(int j = 0; j < combined_population_archive.size(); j++)
        {
            double dist = std::sqrt((combined_population_archive[i]->cached_objectives[0] - combined_population_archive[j]->cached_objectives[0]) * (combined_population_archive[i]->cached_objectives[0] - combined_population_archive[j]->cached_objectives[0]));
            distance_list.push_back(dist);
        }
        sort(distance_list.begin(),distance_list.end());
        double sigma = distance_list[k];
        double D = 1/(sigma+2);
        combined_population_archive[i]->SPEA2_distance = D;
        combined_population_archive[i]->SPEA2_fitness = combined_population_archive[i]->SPEA2_fitness + D;
    }

}

void SPEA2GenerationHandler::SPEA2EnvironmentSelection(std::vector<Node *> & population)
{
    
    //pick up all solutions from archive and population whose SPEA2_fitness smaller than 1
    std::vector<Node *> new_archive;
    for(int i = 0; i < archive.size(); i++)
    {
        if(archive[i]->SPEA2_fitness < 1)
        {
            Node * p = archive[i]->CloneSubtree();
            new_archive.push_back(p);
        }
    }
    for(int i = 0; i< population.size(); i++)
    {
        if(population[i]->SPEA2_fitness < 1)
        {
            Node * p = population[i]->CloneSubtree();
            new_archive.push_back(p);
        }
    }
    
    //if the number of individuals in the new archive is not enough, then supply more individuals  *** now pick new solutions from archive+population BUT IF SUPPLY INDIVIDUALS MUST FROM ARCHIVE?????
    if(new_archive.size() < population.size()) //archive size is always set to the same as population size, if there are too much individuals in archive, then add some dominated solutions (in terms of the fitness calculated by strength and dominance relationship) to the archive 
    {
        std::cout<<"SPEA2EnvironmentSelection flag3-b1: archive size is too smaller than pop size"<<std::endl;
        std::vector<std::pair<int,double>> fitness_list;
        for(int i = 0; i < archive.size()+population.size(); i++)
        {
            if(i<archive.size())
            {
                if(archive[i]->SPEA2_fitness >= 1)
                {
                    std::pair<int,double> p;
                    p.first = i;
                    p.second = archive[i]->SPEA2_fitness;
                    fitness_list.push_back(p);
                }
            }
            else
            {
                if(population[i-archive.size()]->SPEA2_fitness >= 1)
                {
                    std::pair<int,double> p;
                    p.first = i;
                    p.second = population[i-archive.size()]->SPEA2_fitness;
                    fitness_list.push_back(p);
                }
            }
        }

        sort(fitness_list.begin(),fitness_list.end(),compare_pair_int_double);
        int cursor = 0;
        
        while(new_archive.size() < population.size())
        {
            
            int id_real = fitness_list[cursor].first;
            
            if(id_real < archive.size())
            {
                
                Node * p = archive[id_real]->CloneSubtree();
                new_archive.push_back(p);
            }
            else
            {
                
                id_real = id_real - archive.size();
                Node * p = population[id_real]->CloneSubtree();
                new_archive.push_back(p);
            }
            cursor++;
        }
        for(int i = 0; i <archive.size(); i++)
        {
            archive[i]->ClearSubtree();
        }
        archive.clear();
        archive.assign(new_archive.begin(),new_archive.end());
    }
    else//archive size is always set to the same as population size, if there are too much individuals in archive, then delete some of them
    {
        std::cout<<"SPEA2EnvironmentSelection flag3-b2: archive size is too large, size is:"<<new_archive.size()<<std::endl;
        
        std::set<std::vector<double>> uniq_all_inds;
        for(int i = 0; i < new_archive.size(); i++)
        {
            std::vector<double> v;
            v.push_back(new_archive[i]->cached_objectives[0]);
            v.push_back(new_archive[i]->cached_objectives[1]);
            uniq_all_inds.insert(v);
        }
        std::cout<<"altogether "<<new_archive.size()<<" solutions in the archive, while the number of unique solutions is: "<<uniq_all_inds.size()<<std::endl;

        while(new_archive.size() > population.size())
        {
            
            //std::cout<<"        -flag SG-1"<<std::endl;
            vector<vector<double>> distance_list;
            for(int i = 0; i < new_archive.size(); i++){
                vector<double> row;
                distance_list.push_back(row);
            }
            for(int i = 0; i < new_archive.size(); i++){
                for(int j = 0; j < new_archive.size(); j++){
                    if(i == j)
                        continue;
                    double dist = calculate_distance(new_archive[i],new_archive[j]);
                    distance_list[i].push_back(dist);
                    distance_list[j].push_back(dist);
                }
            }
            for(int i = 0; i < new_archive.size(); i++){
                sort(distance_list[i].begin(),distance_list[i].end());
            }


            int neighbor_seq = 0;//for each individual, compare the sparse distance start from the closest individual to the current solution
            std::vector<int> min_dist_inds;
            double min_distance = 1e10;
            for(int j = 0; j < distance_list.size(); j++)
            {
                if(distance_list[j][neighbor_seq] == min_distance){
                    min_dist_inds.push_back(j);
                }
                else if(distance_list[j][neighbor_seq] < min_distance){
                    min_distance = distance_list[j][neighbor_seq];
                    min_dist_inds.clear();
                    min_dist_inds.push_back(j);
                }
            }
            neighbor_seq++;
            std::vector<int> min_dist_inds_copy;
            while(min_dist_inds.size() > 1 && neighbor_seq < distance_list.size()-1) //distance_list[x].size() = distance_list.size()-1
            {
                min_distance = 1e10;
                for(int j = 0; j < min_dist_inds.size(); j++)
                {
                    if(distance_list[min_dist_inds[j]][neighbor_seq] < min_distance)
                    {
                        min_distance = distance_list[min_dist_inds[j]][neighbor_seq];
                        min_dist_inds_copy.clear();
                        min_dist_inds_copy.push_back(min_dist_inds[j]);
                    }
                    else if(distance_list[min_dist_inds[j]][neighbor_seq] == min_distance){
                        min_dist_inds_copy.push_back(min_dist_inds[j]);
                    }
                }
                min_dist_inds.clear();
                min_dist_inds.swap(min_dist_inds_copy);
                neighbor_seq++;
            }
            if(min_dist_inds.size() > 1){
                std::set<std::vector<double>> distance_list_unique;
                for(int i = 0; i < min_dist_inds.size(); i++){
                    distance_list_unique.insert(distance_list[min_dist_inds[i]]);
                }
                if(distance_list_unique.size() != 1)
                {
                    std::runtime_error("warring, more than one solutions exist in min_dist_inds, however, their distance values are not the same, this is wrong!");
                }
            }
            //new_archive[min_dist_inds[0]]->ClearSubtree();
            //new_archive.erase(new_archive.begin()+min_dist_inds[0]);
            
            //to improve the efficiency: if we detect a solution to be delete, then we also check if duplicates of them exist in the new_archive, then we also delete them so that we can avoid to detect them again and again
            std::vector<int> duplicate_list;
            for(int i = 0; i < new_archive.size(); i++)
            {
                bool same = true;
                for(int j = 0; j < new_archive[min_dist_inds[0]]->cached_objectives.size(); j++){
                    if(new_archive[min_dist_inds[0]]->cached_objectives[j] == new_archive[i]->cached_objectives[j]){
                        continue;
                    }else{
                        same = false;
                        break;
                    }
                }
                if(same == true)
                    duplicate_list.push_back(i);
            }
            std::vector<Node *> new_archive_copy;
            if(new_archive.size() >= population.size() + duplicate_list.size())//remove all of the individuals in duplicate_list
            {   
                for(int i = 0; i < new_archive.size(); i++){
                    if(find(duplicate_list.begin(),duplicate_list.end(),i) != duplicate_list.end()){
                        new_archive[i]->ClearSubtree();
                    }else{
                        new_archive_copy.push_back(new_archive[i]);
                    }
                }
            }
            else//remove only some of duplicate solutions in the duplicate_list until archive size equals population size
            {
                int del_num = new_archive.size() - population.size();
                int deleted_num = 0;
                for(int i = 0; i < new_archive.size(); i++){
                    if(deleted_num < del_num && find(duplicate_list.begin(),duplicate_list.end(),i) != duplicate_list.end()){
                        new_archive[i]->ClearSubtree();
                        deleted_num++;
                    }else{
                        new_archive_copy.push_back(new_archive[i]);
                    }
                }
            }
            new_archive.clear();
            new_archive.swap(new_archive_copy);
            new_archive_copy.shrink_to_fit();
        }//end while

        for(int i = 0; i <archive.size(); i++){
                archive[i]->ClearSubtree();
        }
        archive.clear();
        archive.assign(new_archive.begin(),new_archive.end());
        new_archive.clear();
        new_archive.shrink_to_fit();
        
    }
}



void SPEA2GenerationHandler::save_final_population(vector<Node*> & population)//final population is the combination of archive and current population(and only pick up non-dominated solutions)
{
    std::vector<Node *> combined_popu;
    for(int i = 0; i < archive.size(); i++)
    {
        combined_popu.push_back(archive[i]);
    }
    for(int i = 0; i < population.size(); i++)
    {
        combined_popu.push_back(population[i]);
    }
    //std::vector<std::vector<Node *>> fronts = FastNonDominatedSorting(combined_popu);

	//save PF file
	char savefilename[1024];
	char savefilename_archive[1024];
	
    sprintf(savefilename, "./PF/PF_%s_runs%d.txt","SPEA2",conf->num_of_run);
    sprintf(savefilename_archive, "./PF/PF_%s_archive_runs%d.txt","SPEA2",conf->num_of_run);

    //save pf of the last generation
    vector<vector<Node *>> fronts = FastNonDominatedSorting(population);
	std::fstream fout;
	fout.open(savefilename, std::ios::out);
	for(int i = 0; i < fronts[0].size(); i++){
		fout<<fronts[0][i]->cached_objectives[0]<<" "<<fronts[0][i]->cached_objectives[1]<<"\r\n";
	}
	fout.close();
    // fout.open(savefilename, std::ios::out);
	// for(int i = 0; i < population.size(); i++)
	// {
	// 	fout<<population[i]->cached_objectives[0]<<" "<<population[i]->cached_objectives[1]<<"\r\n";
	// }
	// fout.close();


	//save the archive
    //std::vector<std::vector<Node *>> fronts_archive = FastNonDominatedSorting(combined_popu);
	fout.open(savefilename_archive, std::ios::out);
	//for(int i = 0; i < fronts_archive[0].size(); i++){
	//	fout<<fronts_archive[0][i]->cached_objectives[0]<<" "<<fronts_archive[0][i]->cached_objectives[1]<<"\r\n";
	//}
      for(int i = 0; i < archive_output.size(); i++){
        fout<<archive_output[i]->cached_objectives[0]<<" "<<archive_output[i]->cached_objectives[1]<<"\r\n";
    }
	fout.close();

	//copy population in objectives to the external data structure (so that the values can be read by python and calculate the HV value)
	// for(int i = 0; archive.size(); i++)
	// {
	// 	conf->archive_objectives.clear();
	// 	vector<double> ind;
	// 	ind.push_back(archive[i]->cached_objectives[0]);
	// 	ind.push_back(archive[i]->cached_objectives[1]);
	// 	conf->archive_objectives.push_back(ind);

	// }

}




void SPEA2GenerationHandler::update_archive(std::vector<Node *> & population)
{
    std::vector<Node *> all_unique_inds;
	all_unique_inds.reserve(archive_output.size()*5);
	std::set<std::vector<double>> p_unique;
	
	
	std::vector<std::vector<Node *>> layer_population = FastNonDominatedSorting(population);
	
	for(int i = 0; i < layer_population[0].size(); i++)
	{
		std::vector<double> p;
		p.push_back(layer_population[0][i]->cached_objectives[0]);
		p.push_back(layer_population[0][i]->cached_objectives[1]);
		if(p_unique.find(p) == p_unique.end()){
			all_unique_inds.push_back(layer_population[0][i]->CloneSubtree(true));
			p_unique.insert(p);
		}
	}

	for(int i = 0; i < archive_output.size(); i++)
	{
		std::vector<double> p;
		p.push_back(archive_output[i]->cached_objectives[0]);
		p.push_back(archive_output[i]->cached_objectives[1]);
		if(p_unique.find(p) == p_unique.end()){
			all_unique_inds.push_back(archive_output[i]);
			p_unique.insert(p);
		}else{
			archive_output[i]->ClearSubtree(true);
			archive_output[i] = NULL;
		}
	}

	archive_output.clear();
	archive_output.shrink_to_fit();

	vector<vector<Node *>> layers = FastNonDominatedSorting(all_unique_inds);
	for(int i = 0; i < layers[0].size(); i++){
		archive_output.push_back(layers[0][i]);
	}
	for(int i = 1; i < layers.size(); i++){
		for(Node* n : layers[i]){
			n->ClearSubtree();
			n=NULL;
		}
	}
	
	for(int i = 0; i < layers.size(); i++)
	{
		layers[i].clear();
		layers[i].shrink_to_fit();
	}
	layers.clear();
	layers.shrink_to_fit();

	layer_population.clear();
	layer_population.shrink_to_fit();

}

void SPEA2GenerationHandler::PerformGeneration(std::vector<Node *> & population)
{
    std::cout<<"SPEA2 PerformGeneration"<<std::endl;
    fitness->GetPopulationFitness(population, true, conf->caching);
    //std::vector<std::vector<Node *>> front_combined_population = FastNonDominatedSorting(population);

    // fitness assignment
    //fitness->GetPopulationFitness(population, true, conf->caching);//这个函数的fitness指的是各个目标的目标值而非真正意义上的多目标优化的聚合适应度，这是因为这个算法框架最早是求单目标GP构造的，对于多目标GP的一些概念就会含混不清
    
    
    //std::cout<<"real pop size"<<population.size()<<std::endl;
    GetStrength(population);//get strength and thus get the fitness value of each individual

    //Environment selection
    
    SPEA2EnvironmentSelection(population);
     
    //Mating selection:binary tournament selection
    
    vector<vector<double>> distance_table_archive = generate_update_SPEA2_crowding_distance_table();
    int pop_size = archive.size();

	vector<Node*> selected_parents; selected_parents.reserve(population.size());
	while(selected_parents.size() < pop_size) {
		selected_parents.push_back(SPEA2TournamentSelectionWinner(population, 2,distance_table_archive));//in SPEA2, tournament selection size is always set to 2
	}
    
     // Variation
    std::vector<Node*> offspring = MakeOffspring(archive, selected_parents);
    
    // Update fitness
    fitness->GetPopulationFitness(offspring, true, conf->caching);
    
    for(int i = 0; i < population.size(); i++)
    {
        population[i]->ClearSubtree();
    }
    population.clear();
    population = offspring;
    
    
    update_archive(population);

    //std::cout<<"SPEA2 flag8"<<std::endl;
    //save PF file and archive file during the iteration
	char savefilename[1024];
	sprintf(savefilename, "./PMF/PMF_gen%d.txt",conf->current_generation);
	std::fstream fout;
	fout.open(savefilename, std::ios::out);
	for(int i = 0; i < population.size(); i++)
	{
		fout<<population[i]->cached_objectives[0]<<" "<<population[i]->cached_objectives[1]<<"\r\n";
	}
	fout.close();
    sprintf(savefilename, "./PMF_archive/archive_gen%d.txt",conf->current_generation);
    fout.open(savefilename, std::ios::out);
    for(int i = 0; i < archive_output.size(); i++){
        fout<<archive_output[i]->cached_objectives[0]<<" "<<archive_output[i]->cached_objectives[1]<<"\r\n";
    }
    fout.close();

    FastNonDominatedSorting(population);
    //Termination
    //the final output is the combination of current population and archive

    malloc_trim(0);
    
    if(conf->current_generation == conf->max_generations)
    {
        std::cout<<"save final population... ...";
        save_final_population(population);
        std::cout<<"finished!"<<std::endl;
        std::cout<<"cleaning individuals...";
        for(int i = 0; i < archive.size(); i++){
            archive[i]->ClearSubtree();
        }
        std::cout<<"finished! REDAY TO FINISH THE algorithm"<<std::endl;
        FastNonDominatedSorting(population);
        return;
    }
}

vector<vector<double>> SPEA2GenerationHandler::generate_update_SPEA2_crowding_distance_table()
{
    vector<vector<double>> dist_table;
     for(int i = 0; i < archive.size(); i++)
    {
        vector<double> row;
        for(int j = 0; j < archive.size(); j++)
        {
            row.push_back(0);
        }
        dist_table.push_back(row);
    }
    for(int i = 0; i < archive.size(); i++)
    {
        for(int j = 0; j < archive.size(); j++)
        {
            if(i == j) continue;
            double dist = calculate_distance(archive[i],archive[j]);
            dist_table[i][j] = dist;
            dist_table[j][i] = dist;
        }
    }

    return dist_table;
}

Node * SPEA2GenerationHandler::SPEA2TournamentSelectionWinner(const std::vector<Node*>& candidates, size_t tournament_size, vector<vector<double>> dist_table) 
{
    int idx1 = arma::randu() * candidates.size();
    Node * winner = candidates[ idx1 ];
    int idx2 = -1;
    for (size_t i = 1; i < tournament_size; i++) 
    {
        idx2 = arma::randu() * candidates.size();
        Node * candidate = candidates[ idx2 ];
        if (candidate->SPEA2_fitness < winner->SPEA2_fitness) 
        {
            winner = candidate;
        }
        else if(candidate->SPEA2_fitness == winner->SPEA2_fitness)
        {
            for(int i = 0; i < archive.size(); i++)
            {
                if(dist_table[idx1][i] < dist_table[idx2][i])
                    winner = candidate;
            }
        }
    }
    
    return winner;
}

