#include "GPGOMEA/Evolution/AlphaDominance.h"

#include <malloc.h>

using namespace std;
using namespace arma;

void AlphaDominance::adaptive_update_alpha(std::vector<Node *> & population)
{
	if(conf->current_generation <= 1)
	{
		double u_eff = 0;
		double l_eff = 99999;
		int u_size = 0;
		int l_size = 99999;
		current_alpha = 0;
		return;
	}


	for(int i = 0; i < population.size(); i++)
	{
		//obj1: MSE
		//obj2: exp length
		if(population[i]->cached_objectives[0] > u_eff)
			u_eff = population[i]->cached_objectives[0];
		if(population[i]->cached_objectives[0] < l_eff)
			l_eff = population[i]->cached_objectives[0];
		if(population[i]->cached_objectives[1] > u_size)
			u_size = population[i]->cached_objectives[1];
		if(population[i]->cached_objectives[1] < l_size)
			l_size = population[i]->cached_objectives[1];	
	}
	vector<vector<Node *>> fronts = FastNonDominatedSorting(population);
	double average_fitness = 0;
	double average_size = 0;
	for(int i = 0; i < fronts[0].size(); i++)
	{
		average_fitness += fronts[0][i]->cached_objectives[0];
		average_size += fronts[0][i]->cached_objectives[1];
	}
	average_fitness /= fronts[0].size();
	average_size /= fronts[0].size();

	if( (u_eff -average_fitness)/(average_fitness - l_eff) > 1 && (u_size - average_size)/(average_size - l_size) < 1)
	{
		current_alpha = current_alpha - 0.2; //0.2 is the lr, the learning rate in adaptive alpha-dominance
		current_alpha = current_alpha < 0 ? 0 : current_alpha;
		std::cout<<"	currently the alpha is:"<<current_alpha<<std::endl;
	}
	else if( (u_eff -average_fitness)/(average_fitness - l_eff) < 1 && (u_size - average_size)/(average_size - l_size) > 1 )
	{
		current_alpha = current_alpha + 0.2;
		std::cout<<"	currently the alpha is:"<<current_alpha<<std::endl;
	}
	else
	{
		std::cout<<"alpha doesn't change"<<std::endl;
	} 

}


void AlphaDominance::get_alpha()
 {
	dominance_alpha.clear();
	if(conf->methodtype != "alpha_dominance")
	{
		std::cout<<"the method is: "<<conf->methodtype<<" which doesn't match with alpha_dominance. If you want to use alpha dominance, please set the methodtype to <alpha_dominance>"<<std::endl;
		exit(-1);
	}
	int param_C = 99999999;
	double alpha_exp = 0;
	double alpha_mse = 0;
	if(conf->alphafunction == "linear")
	{
		alpha_exp = param_C + (-param_C * conf->current_generation)/conf->max_generations;
	}
	else if(conf->alphafunction == "sigmoid")
	{
		double val1 = std::exp(conf->current_generation - conf->max_generations/2);
		double val2 = 1/(1+val1);
		alpha_exp = param_C * val2;
	}
	else if(conf->alphafunction == "cosine")
	{
		double val = (3.1415926*double(conf->current_generation))/10;
		double cos_val = param_C / 2 * (std::cos(val) + 1);
		alpha_exp = cos_val;
	}
	else
	{
		std::cout<<"error! the methodtype is set to alpha_dominance, but the alpha function is set to: "<<conf->alphafunction<<" which is not linear/sigmoid/cosin. Please set the correct parameter!"<<std::endl;
		exit(-1);
	}
	//the first obj is mse, the second obj is exp length
	dominance_alpha.push_back(alpha_mse);
	dominance_alpha.push_back(alpha_exp);
	
 }




std::vector<std::vector<Node*>> AlphaDominance::FastNonDominatedSorting_alpha_dominance(std::vector<Node*> & population){
	
	double vm=0, rss=0;
    process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance VM: " << vm << "; RSS: " << rss << std::endl;
	size_t rank_counter = 0;
	vector<vector<Node*>> nondominated_fronts; 
	nondominated_fronts.reserve(20);
	unordered_map<Node*, std::vector<Node*>> dominated_solutions;
	unordered_map<Node*, int> domination_counts;
	//map<Node*, std::vector<Node*>> dominated_solutions;
	//map<Node*, int> domination_counts;
	vector<Node*> current_front; current_front.reserve(population.size() / 2);


	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag1 VM: " << vm << "; RSS: " << rss << std::endl;

	for (size_t i = 0; i < population.size(); i++) {
		Node * p = population[i];

		dominated_solutions[p].reserve(population.size() / 2);
		domination_counts[p] = 0;

		for(size_t j = 0; j < population.size(); j++) {
			if (i == j)
				continue;
			Node * q = population[j];

			if(p->Dominates(q,dominance_alpha[0],dominance_alpha[1]))
				dominated_solutions[p].push_back(q);
			else if (q->Dominates(p,dominance_alpha[0],dominance_alpha[1]))
				domination_counts[p]++;
		}

		if (domination_counts[p] == 0) {
			p->rank = rank_counter;
			current_front.push_back(p);
		}

	}

	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag2 VM: " << vm << "; RSS: " << rss << std::endl;


	while(current_front.size() > 0) {
		vector<Node*> next_front; next_front.reserve(population.size() / 2);
		for(Node * p : current_front) {
			for(Node * q : dominated_solutions[p]) {
				domination_counts[q] -= 1;
				if (domination_counts[q] == 0) {
					q->rank = rank_counter + 1;
					next_front.push_back(q);
				}
			}
		}
		nondominated_fronts.push_back(current_front);
		rank_counter++;
		current_front = next_front;
		
		next_front.clear();
		next_front.shrink_to_fit();
	}

	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag3 VM: " << vm << "; RSS: " << rss << std::endl;

	for(auto it = dominated_solutions.begin(); it != dominated_solutions.end();it++)
	{
		it->second.clear();
		it->second.shrink_to_fit();
	}

	dominated_solutions.clear();//dominated_solutions.rehash(0);
	domination_counts.clear();//domination_counts.rehash(0);

	domination_counts.rehash(0);
	dominated_solutions.rehash(0);

	//std::map<Node*, std::vector<Node*>>().swap(dominated_solutions);
	//std::map<Node*,int>().swap(domination_counts);

	current_front.clear();
	current_front.shrink_to_fit();

	std::cout<<"number of layers:"<<nondominated_fronts.size()<<std::endl;
	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag4 VM: " << vm << "; RSS: " << rss << std::endl;

	return nondominated_fronts;
}






std::vector<std::vector<Node*>> AlphaDominance::FastNonDominatedSorting_adaptive_alpha_dominance(std::vector<Node*> & population)
{
	double vm=0, rss=0;
    process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance VM: " << vm << "; RSS: " << rss << std::endl;
	size_t rank_counter = 0;
	vector<vector<Node*>> nondominated_fronts; 
	nondominated_fronts.reserve(10);
	unordered_map<Node*, std::vector<Node*>> dominated_solutions;
	unordered_map<Node*, int> domination_counts;
	vector<Node*> current_front; 
	current_front.reserve(population.size() / 2);

	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag1 VM: " << vm << "; RSS: " << rss << std::endl;

	for (size_t i = 0; i < population.size(); i++) {
		Node * p = population[i];

		dominated_solutions[p].reserve(population.size() / 2);
		domination_counts[p] = 0;

		for(size_t j = 0; j < population.size(); j++) {
			if (i == j)
				continue;
			Node * q = population[j];

			if(p->Dominates(q,current_alpha,current_alpha))
				dominated_solutions[p].push_back(q);
			else if (q->Dominates(p,current_alpha,current_alpha))
				domination_counts[p]++;
		}

		if (domination_counts[p] == 0) {
			p->rank = rank_counter;
			current_front.push_back(p);
		}
	}

	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag2 VM: " << vm << "; RSS: " << rss << std::endl;

	while(current_front.size() > 0) {
		vector<Node*> next_front; next_front.reserve(population.size() / 2);
		for(Node * p : current_front) {
			for(Node * q : dominated_solutions[p]) {
				domination_counts[q] -= 1;
				if (domination_counts[q] == 0) {
					q->rank = rank_counter + 1;
					next_front.push_back(q);
				}
			}
		}
		nondominated_fronts.push_back(current_front);
		rank_counter++;
		current_front = next_front;
		//current_front.swap(next_front);
		
	}
	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag3 VM: " << vm << "; RSS: " << rss << std::endl;

	return nondominated_fronts;

}



void AlphaDominance::PerformGeneration_alpha_dominance(std::vector<Node*> & population) {
	
	double elitism_rate = 0.01;	
	double elitism_num = population.size()*elitism_rate;

	size_t pop_size = population.size();

	std::vector<std::vector<Node*>> fronts =FastNonDominatedSorting_alpha_dominance(population);
	//std::vector<std::vector<Node*>> fronts =FastNonDominatedSorting(population);
	for(int i = 0; i < fronts.size(); i++){
		ComputeCrowndingDistance(fronts[i]);
	}
	vector<Node*> selected_parents; 
	while(selected_parents.size() < pop_size) {
		selected_parents.push_back(TournamentSelection::GetMOTournamentSelectionWinner(population, conf->tournament_selection_size));
	}

	int elitism_size = -99999;
	if(archive.size() <= elitism_num) // elitism size is set to 1% of the population size
		elitism_size = archive.size();
	else
		elitism_size = elitism_num;
	if(elitism_size == 0)
		elitism_size = 1;
	int number_individuals_transfer_to_population = 0;
	if(archive.size() <= elitism_size)
		number_individuals_transfer_to_population = archive.size();
	else
		number_individuals_transfer_to_population = elitism_size;

	std::vector<Node*> offspring = MakeOffspring(population, selected_parents);

	for(int i = 0; i < population.size(); i++){
		population[i]->ClearSubtree(false);
		population[i] = nullptr;
	}
	population.clear();
	//population.shrink_to_fit();
	//population.reserve(offspring.size());
	//population = offspring;
	//offspring.clear();
	//offspring.shrink_to_fit();
	population.insert(population.end(),offspring.begin(),offspring.end());

	if(number_individuals_transfer_to_population > 0)
	{
		std::set<int> idx_delete;
		while(idx_delete.size() < number_individuals_transfer_to_population){
			idx_delete.insert(population.size() * arma::randu());
		}
		for(std::set<int>::iterator it = idx_delete.begin(); it != idx_delete.end(); it++){
			population[*it]->ClearSubtree(true);
			population[*it] = nullptr;
		}
		for(std::vector<Node *>::iterator it = population.begin(); it != population.end(); ){
			if(*it == nullptr){
				it = population.erase(it);
			}else{
				it++;
			}
		}
	}

	if(archive.size() <= elitism_size){
		for(int i = 0; i < archive.size(); i++){
			population.push_back(archive[i]->CloneSubtree(true));
		}
	}else{
		set<int> id_set;
		while(id_set.size()< elitism_size)
		{
			int idx = archive.size() * arma::randu();
			id_set.insert(idx);
		}for(std::set<int>::iterator it = id_set.begin(); it != id_set.end(); it++)
		
		{
			population.push_back(archive[*it]->CloneSubtree(true));
		}
	}
	 // Update fitness
    fitness->GetPopulationFitness(population, true, conf->caching);

	population.shrink_to_fit();

	//save PF file
	char savefilename[1024];
	sprintf(savefilename, "./PMF/PMF_gen%d.txt",conf->current_generation);
	std::fstream fout;
	fout.open(savefilename, std::ios::out);

	for(int i = 0; i < population.size(); i++)
	{
		fout<<population[i]->cached_objectives[0]<<" "<<population[i]->cached_objectives[1]<<"\r\n";
	}
	fout.close();

	update_archive(population);

	sprintf(savefilename, "./PMF_archive/archive_gen%d.txt",conf->current_generation);
    fout.open(savefilename, std::ios::out);
    for(int i = 0; i < archive.size(); i++)
    {
        fout<<archive[i]->cached_objectives[0]<<" "<<archive[i]->cached_objectives[1]<<"\r\n";
    }

    fout.close();
	malloc_trim(0);
	
	if(conf->current_generation == conf->max_generations)
	{
		save_final_population(population);
		
		for (Node * n : archive) 
		{
			n->ClearSubtree(true);
		}
		archive.clear();
		archive.shrink_to_fit();
	}

}





void AlphaDominance::PerformGeneration_adaptive_alpha_dominance(std::vector<Node *> & population)
{
	double elitism_rate = 0.01;
	std::cout<<"PerformGeneration_adaptive_alpha_dominance"<<std::endl;

	// Keep track of normal pop_size
	size_t pop_size = population.size();

	std::vector<std::vector<Node*>> fronts = FastNonDominatedSorting_adaptive_alpha_dominance(population);
	for(int i = 0; i < fronts.size(); i++)
	{
		ComputeCrowndingDistance(fronts[i]);
	}

	adaptive_update_alpha(population);

	// Parent selection based on MO tournaments
	vector<Node*> selected_parents; 
	selected_parents.reserve(population.size());
	while(selected_parents.size() < pop_size) {
		selected_parents.push_back(TournamentSelection::GetMOTournamentSelectionWinner(population, conf->tournament_selection_size));
	}


    // Variation
    //std::vector<Node*> offspring = MakeOffspring(population, selected_parents);
	
	int elitism_size = -99999;
	if(archive.size() <= population.size()*elitism_rate) // elitism size is set to 1% of the population size
		elitism_size = archive.size();
	else
		elitism_size = population.size()*elitism_rate;
	if(elitism_size == 0)
		elitism_size = 1;
	
	//std::cout<<"	elitism size(target):"<<elitism_size<<std::endl;

	//std::cout<<"	population size"<<population.size()<<std::endl;
	//std::cout<<"	selected_parents size"<<selected_parents.size()<<std::endl;
	//std::cout<<"	elitism_size size"<<elitism_size<<std::endl;
	int number_individuals_transfer_to_population = 0;
	if(archive.size() <= elitism_size)
		number_individuals_transfer_to_population = archive.size();
	else
		number_individuals_transfer_to_population = elitism_size;
	
	//std::vector<Node*> offspring = MakeOffspring_exempt_elitism(population, selected_parents,number_individuals_transfer_to_population);
	std::vector<Node*> offspring = MakeOffspring(population, selected_parents);
	
	//randomly delete some solutions from the population so that there are slots for archive individuals
	if(number_individuals_transfer_to_population != 0)
	{
		std::set<int> idx_delete;
		while(idx_delete.size() < number_individuals_transfer_to_population)
		{
			idx_delete.insert(offspring.size() * arma::randu());
		}
		std::vector<Node *> offspring_pruning;
		for(int i = 0; i < population.size(); i++)
		{
			if(idx_delete.find(i)!=idx_delete.end())
			{
				offspring[i]->ClearSubtree();
				continue;
			}
			offspring_pruning.push_back(offspring[i]);
		}
		offspring.clear();
		offspring = offspring_pruning;
	}
	std::cout<<"	after making offspring and reserve slots for archive individual, the offspring size is:"<<offspring.size()<<std::endl;
	
	
	if(archive.size() <= elitism_size)//archive中的个体数量达不到
	{
		for(int i = 0; i < archive.size(); i++)
		{
			Node * p;
			p = archive[i]->CloneSubtree();
			offspring.push_back(p);
		}
		std::cout<<"	after transfer indivdiuals in archive to the current offspring population, the offspring size is:"<<offspring.size()<<std::endl;
	}
	else//archive中的非支配个体数量很多，则随机选择一部分迁移到种群中
	{
		set<int> id_set;
		while(id_set.size()< elitism_size)
		{
			int idx = archive.size() * arma::randu();
			id_set.insert(idx);
		}
		for(std::set<int>::iterator it = id_set.begin(); it != id_set.end(); it++)
		{
			Node *p = archive[*it]->CloneSubtree();
			offspring.push_back(p);
		}
	}
	std::cout<<"	After insert archive elements to offspring list, the size:"<<offspring.size()<<std::endl;	


    // Update fitness
    fitness->GetPopulationFitness(offspring, true, conf->caching);

	/*
    for(size_t i=0; i < fronts.size(); i++) 
	{
    	for (Node * n : fronts[i]) 
		{
    		n->ClearSubtree();
    	}
    }
	*/

	for (Node * n : population) {
		n->ClearSubtree();
	}
    population.clear();
	//population.assign(offspring.begin(), offspring.end());
	population = offspring;


	//save PF file
	char savefilename[1024];
	sprintf(savefilename, "./PMF/PMF_gen%d.txt",conf->current_generation);
	std::fstream fout;
	fout.open(savefilename, std::ios::out);

	for(int i = 0; i < population.size(); i++)
	{
		fout<<population[i]->cached_objectives[0]<<" "<<population[i]->cached_objectives[1]<<"\r\n";
	}
	fout.close();

	update_archive(population);
	
	sprintf(savefilename, "./PMF_archive/archive_gen%d.txt",conf->current_generation);
    fout.open(savefilename, std::ios::out);
    for(int i = 0; i < archive.size(); i++)
    {
        fout<<archive[i]->cached_objectives[0]<<" "<<archive[i]->cached_objectives[1]<<"\r\n";
    }
    fout.close();

		
	malloc_trim(0);

	if(conf->current_generation == conf->max_generations)
	{
		save_final_population(population);
		
		for (Node * n : archive) 
		{
			n->ClearSubtree();
		}
	}
}


void AlphaDominance::PerformGeneration(std::vector<Node*> & population)
{

	std::cout<<"	in AlphaDominance/AdaptiveAlphaDominance, current gen="<<conf->current_generation<<std::endl;

	double vm=0;
	double rss=0;
    process_mem_usage(vm, rss); 
	std::cout << "At the start of this iteration, VM: " << vm << "; RSS: " << rss << std::endl;

	if(conf->methodtype == "alpha_dominance"){
		get_alpha();	
		PerformGeneration_alpha_dominance(population);
	}
	else if(conf->methodtype == "adaptive_alpha_dominance"){
		PerformGeneration_adaptive_alpha_dominance(population);
	}
	else{
		throw std::runtime_error("the selected method should be adaptive alpha dominance or alpha dominance, error");
	}
}


