#include "GPGOMEA/Evolution/NSGA2GenerationHandler.h"



#include <cstdlib>

#include <malloc.h>

using namespace std;
using namespace arma;


void process_mem_usage(double & vm_usage, double & resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

struct compare_pair_double_double_for_set{
	bool operator()(const pair<double,double> & x, const pair<double,double> & y) const{
		if(x.first < y.first || x.first == y.first && x.second < y.second )
			return true;
		else 
			return false;
	}
};

struct cmpKey_set_vectorDOUBLE{
	bool operator()(const vector<double> & elem1, const vector<double> & elem2) const{
		if(elem1.size() != elem2.size())
			std::cout<<"error set<vector<double>> comp operator error, length of two comp elem in different size"<<std::endl;
		for(int i = 0; i <elem1.size(); i++)
		{
			if(elem1[i] < elem2[i])
				return true;
		}
		return false;
	}
};


void NSGA2GenerationHandler::MakeOffspring_NSGA2GENERATIONHANDLER(const std::vector<Node *> & population, const std::vector<Node*> & selected_parents, std::vector<Node *> & offspring) {

    std::cout<<"    go into MakeOffspring function"<<std::endl;

	offspring.clear();
	offspring.shrink_to_fit();
	offspring.assign(population.size(),nullptr);
   // std::vector<Node *> offspring(population.size(), NULL);


 // Variation
    size_t offspring_size, variator_limit;
    
	size_t offspring_size_pvt = 0;
	size_t variator_limit_pvt = population.size() * conf->subtree_crossover_proportion;
	size_t population_chunk, population_chunk_start_idx, max_attempts_variation, attempts_variation;

	max_attempts_variation = 10;
	attempts_variation = 0;

	population_chunk = (variator_limit_pvt / omp_get_num_threads());
	population_chunk_start_idx = population_chunk * omp_get_thread_num();

	// crossover
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

    assert(offspring.size() == population.size());
    std::cout<<"    return from MakeOffspring function"<<std::endl;
    //return offspring;
}






void NSGA2GenerationHandler::update_archive(std::vector<Node *> & population)
{
	
	std::cout<<"before add new elements, archive size: "<<archive.size()<<std::endl;
	
	//individuals come from two sources: 1. current population, and 2. archive of the last generation
	//firstly, combine population and archive together and unique filter them
	std::vector<Node *> all_unique_inds;
	all_unique_inds.reserve(archive.size()*10);
	std::set<std::vector<double>> p_unique;
	
	
	std::vector<std::vector<Node *>> layer_population = FastNonDominatedSorting(population);
	std::cout<<"1st layer of the population contains "<<layer_population[0].size()<<" individuals"<<std::endl;
	
	std::cout<<"get individuals from 1st layer of population"<<std::endl;
	for(int i = 0; i < layer_population[0].size(); i++)
	{
		std::vector<double> p;
		p.push_back(layer_population[0][i]->cached_objectives[0]);
		p.push_back(layer_population[0][i]->cached_objectives[1]);
		if(p_unique.find(p) == p_unique.end()){
			std::cout<<"detect new ind:"<<p[0]<<","<<p[1];
			all_unique_inds.push_back(layer_population[0][i]->CloneSubtree(true));
			p_unique.insert(p);
			std::cout<<", finish record the item"<<std::endl;
		}
	}

	std::cout<<"after extract individuals from 1st layer of current population, the p_unique size is:"<<p_unique.size()<<", and the all_unique_inds size is:"<<all_unique_inds.size()<<std::endl;


	std::cout<<"get individuals from archive"<<std::endl;
	for(int i = 0; i < archive.size(); i++)
	{
		std::vector<double> p;
		p.push_back(archive[i]->cached_objectives[0]);
		p.push_back(archive[i]->cached_objectives[1]);
		if(p_unique.find(p) == p_unique.end()){
			all_unique_inds.push_back(archive[i]);
			p_unique.insert(p);
		}else{
			archive[i]->ClearSubtree(true);
			archive[i] = NULL;
		}
	}

	std::cout<<"after extract individuals from current archive, the p_unique size is:"<<p_unique.size()<<", and the all_unique_inds size is:"<<all_unique_inds.size()<<std::endl;
	//std::cout<<"after combine population and archive, the number of unique solutions: "<<all_unique_inds.size();
	archive.clear();
	archive.shrink_to_fit();

	vector<vector<Node *>> layers = FastNonDominatedSorting(all_unique_inds);
	for(int i = 0; i < layers[0].size(); i++){
		archive.push_back(layers[0][i]);
	}
	for(int i = 1; i < layers.size(); i++){
		for(Node* n : layers[i]){
			n->ClearSubtree(true);
			n=NULL;
		}
	}
	
	std::cout<<" ,after nondominated sorting, the archive size is:"<<archive.size();

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


void NSGA2GenerationHandler::save_final_population(vector<Node*> & population)
{
	//save PF file
	char savefilename[1024];
	char savefilename_archive[1024];
	if(conf->algorithm_framework == "NSGA2" && conf->methodtype == "alpha_dominance")
	{
		char buf1[128];
		char buf2[128];
		strcpy(buf1,(conf->methodtype).c_str());
		strcpy(buf2,(conf->alphafunction).c_str());
		sprintf(savefilename, "./PF/PF_%s_%s_runs%d.txt",buf1,buf2,conf->num_of_run);
		sprintf(savefilename_archive, "./PF/PF_%s_%s_archive_runs%d.txt",buf1,buf2,conf->num_of_run);
	}
	else if(conf->algorithm_framework == "NSGA2" && conf->methodtype == "adaptive_alpha_dominance" || conf->algorithm_framework == "NSGA2DP" || conf->algorithm_framework == "NSGA2" && conf->methodtype == "classic")
	{
		char buf1[128];
		strcpy(buf1,(conf->methodtype).c_str());
		sprintf(savefilename, "./PF/PF_%s_runs%d.txt",buf1,conf->num_of_run);
		sprintf(savefilename_archive, "./PF/PF_%s_archive_runs%d.txt",buf1,conf->num_of_run);
	}
	else
	{
		std::cout<<"ERROR THE methodtype is:"<<conf->methodtype<<", which hasn't been defined in the algorithm!"<<std::endl;
		exit(-1);
	}

	std::vector<std::vector<Node *>> fronts = FastNonDominatedSorting(population);


	//std::cout<<savefilename<<std::endl;
	std::fstream fout;
	fout.open(savefilename, std::ios::out);
	for(int i = 0; i < fronts[0].size(); i++)
	{
		fout<<population[i]->cached_objectives[0]<<" "<<population[i]->cached_objectives[1]<<"\r\n";
	}
	fout.close();


	//save the archive
	fout.open(savefilename_archive, std::ios::out);
	for(int i = 0; i < archive.size(); i++)
	{
		fout<<archive[i]->cached_objectives[0]<<" "<<archive[i]->cached_objectives[1]<<"\r\n";
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








// ORIGINAL VERSION OF NSGAII IN MARCO'S VERSION
void NSGA2GenerationHandler::PerformGeneration_ORI(std::vector<Node*> & population) {
	
	std::cout<<"this is the original version of PerformGeneration (NSGA2) CTRL+C from Marco's github"<<std::endl;
	
	// Keep track of normal pop_size
	size_t pop_size = population.size();

	

	// Parent selection based on MO tournaments
	vector<Node*> selected_parents; selected_parents.reserve(population.size());
	while(selected_parents.size() < pop_size) {
		selected_parents.push_back(TournamentSelection::GetMOTournamentSelectionWinner(population, conf->tournament_selection_size));
	}

    // Variation
    vector<Node*> offspring = MakeOffspring(population, selected_parents);

    // Update fitness
    fitness->GetPopulationFitness(offspring, true, conf->caching);

    // P+O
    population.insert(population.end(), offspring.begin(), offspring.end());

    // Assign ranks
    //vector<vector<Node*>> fronts = FastNonDominatedSorting(population);
	//std::cout<<"the number of fronts (when sorting based on normal dominance based NDSort):"<<fronts.size()<<std::endl;

	//vector<vector<Node*>> fronts = FastNonDominatedSorting_alpha_dominance(population);
	vector<vector<Node*>> fronts = FastNonDominatedSorting(population);
	std::cout<<"the number of fronts (when sorting based on alpha-dominance based NDSort):"<<fronts.size()<<std::endl;


	// Pick survivors
    vector<Node*> selection; selection.reserve(pop_size);
    size_t current_front_idx = 0;
    while ( current_front_idx < fronts.size() && 
    	fronts[current_front_idx].size() + selection.size() < pop_size ) {
    	
    	ComputeCrowndingDistance(fronts[current_front_idx]);
    	selection.insert(selection.end(), fronts[current_front_idx].begin(), fronts[current_front_idx].end());
    	current_front_idx++;
    }

    // insert remaining solutions
    if (selection.size() < pop_size) {
    	ComputeCrowndingDistance(fronts[current_front_idx]);
    	sort( fronts[current_front_idx].begin(), fronts[current_front_idx].end(), [](const Node * lhs, const Node * rhs) 
	    	{
	    		return lhs->crowding_distance > rhs->crowding_distance;
	    	});

    	while(selection.size() < pop_size) {
    		selection.push_back(fronts[current_front_idx][0]);
    		fronts[current_front_idx].erase(fronts[current_front_idx].begin()); // remove first element
    	}
    }

    // cleanup leftovers
    for(size_t i=current_front_idx; i < fronts.size(); i++) {
    	for (Node * n : fronts[i]) {
    		n->ClearSubtree();
    	}
    }

    // Update population
    population = selection;

}


void NSGA2GenerationHandler::PerformGeneration(std::vector<Node*> & population) {

	std::cout<<"	in NSGAII, current gen="<<conf->current_generation<<std::endl;

	double vm=0;
	double rss=0;
    process_mem_usage(vm, rss); 
	std::cout << "At the start of this iteration, VM: " << vm << "; RSS: " << rss << std::endl;

	if(conf->methodtype == "alpha_dominance")
	{
		throw std::runtime_error("error, the ada-alpha-dominance and alpha-dominance based methods are not in NSGA2GenerationHandler.cpp, please check if the parameter is correctly set up");
	}
	else if(conf->methodtype == "adaptive_alpha_dominance")
	{
		throw std::runtime_error("error, the ada-alpha-dominance and alpha-dominance based methods are not in NSGA2GenerationHandler.cpp, please check if the parameter is correctly set up");
	}
	else if(conf->methodtype == "Duplicate_control")
	{
		//PerformGeneration_Non_duplicate(population);
		throw std::runtime_error("error, NSGA2DP is not implemented in NSGAII base class, the code of calling NSGA2PD should be checked");
	}
	
	//std::cout<<"The selected version is: "<<conf->methodtype<<" ,execute normal NSGA-II process"<<std::endl;
	
    process_mem_usage(vm, rss); std::cout << "after selecting which version of performgeneration executed, VM: " << vm << "; RSS: " << rss << std::endl;

	// Keep track of normal pop_size
	size_t pop_size = population.size();

	// Parent selection based on MO tournaments
	vector<Node*> selected_parents; selected_parents.reserve(population.size());
	while(selected_parents.size() < pop_size) {
		selected_parents.push_back(TournamentSelection::GetMOTournamentSelectionWinner(population, conf->tournament_selection_size));
	}

    process_mem_usage(vm, rss); std::cout << "after tournamentselection, VM: " << vm << "; RSS: " << rss << std::endl;

    // Variation
    std::vector<Node*> offspring = MakeOffspring(population, selected_parents);

	process_mem_usage(vm, rss); std::cout << "after variation, VM: " << vm << "; RSS: " << rss << std::endl;

    // Update fitness
    fitness->GetPopulationFitness(offspring, true, conf->caching);

	process_mem_usage(vm, rss); std::cout << "after updating fitness, VM: " << vm << "; RSS: " << rss << std::endl;

    // P+O
    population.insert(population.end(), offspring.begin(), offspring.end());

	process_mem_usage(vm, rss); std::cout << "after P+O, VM: " << vm << "; RSS: " << rss << std::endl;

	std::vector<std::vector<Node*>> fronts;

    // Assign ranks
	fronts = FastNonDominatedSorting(population);

	process_mem_usage(vm, rss); std::cout << "after fast non-dominated sorting, VM: " << vm << "; RSS: " << rss << std::endl;

	// Pick survivors
    vector<Node*> selection; selection.reserve(pop_size);
    size_t current_front_idx = 0;
    while ( current_front_idx < fronts.size() && 
    	fronts[current_front_idx].size() + selection.size() < pop_size ) {
    	
    	ComputeCrowndingDistance(fronts[current_front_idx]);
    	selection.insert(selection.end(), fronts[current_front_idx].begin(), fronts[current_front_idx].end());
    	current_front_idx++;
    }

	process_mem_usage(vm, rss); std::cout << "after pickup survivors, VM: " << vm << "; RSS: " << rss << std::endl;

    // insert remaining solutions
    if (selection.size() < pop_size) {
    	ComputeCrowndingDistance(fronts[current_front_idx]);
    	sort( fronts[current_front_idx].begin(), fronts[current_front_idx].end(), [](const Node * lhs, const Node * rhs) 
	    	{
	    		return lhs->crowding_distance > rhs->crowding_distance;
	    	});

    	while(selection.size() < pop_size) {
    		selection.push_back(fronts[current_front_idx][0]);
    		fronts[current_front_idx].erase(fronts[current_front_idx].begin()); // remove first element
    	}
    }

	process_mem_usage(vm, rss); std::cout << "after inserting remaining solutions, VM: " << vm << "; RSS: " << rss << std::endl;

    // cleanup leftovers
    for(size_t i=current_front_idx; i < fronts.size(); i++) {
    	for (Node * n : fronts[i]) {
    		n->ClearSubtree();
    	}
    }

	process_mem_usage(vm, rss); std::cout << "after cleanup leftovers, VM: " << vm << "; RSS: " << rss << std::endl;

    // Update population
    population = selection;

	//update archive
	vector<Node *> new_archive;
	for(int i = 0; i < population.size(); i++)
	{
		new_archive.push_back(population[i]);
	}
	for(int j = 0; j < archive.size(); j++)
	{
		new_archive.push_back(archive[j]);
	}
	
	
	std::vector<vector<Node *>> fronts_new_archive = FastNonDominatedSorting(new_archive);
	new_archive.clear();
	for(int i = 0; i < fronts_new_archive[0].size(); i++)
	{
		Node *n = fronts_new_archive[0][i]->CloneSubtree();
		new_archive.push_back(n);
	}
	for(int i = 0; i < archive.size(); i++)
	{
		archive[i]->ClearSubtree();
	}
	archive.clear();
	archive.assign(new_archive.begin(),new_archive.end());
	
	process_mem_usage(vm, rss); std::cout << "after update archive, VM: " << vm << "; RSS: " << rss << std::endl;


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
	

	

	sprintf(savefilename, "./PMF_archive/archive_gen%d.txt",conf->current_generation);
    fout.open(savefilename, std::ios::out);
    for(int i = 0; i < archive.size(); i++)
    {
        fout<<archive[i]->cached_objectives[0]<<" "<<archive[i]->cached_objectives[1]<<"\r\n";
    }
    fout.close();

	process_mem_usage(vm, rss); std::cout << "after save file to hdd (finish this iteration), VM: " << vm << "; RSS: " << rss << std::endl;

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





std::vector<std::vector<Node*>> NSGA2GenerationHandler::FastNonDominatedSorting(std::vector<Node*> & population){
	
	double vm;
	double rss;
	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance VM: " << vm << "; RSS: " << rss << std::endl;
	
	size_t rank_counter = 0;
	vector<vector<Node*>> nondominated_fronts; 
	//nondominated_fronts.reserve(10);
	unordered_map<Node*, std::vector<Node*>> dominated_solutions;
	unordered_map<Node*, int> domination_counts;
	vector<Node*> current_front; 
	//current_front.reserve(population.size() / 2);

	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag1 VM: " << vm << "; RSS: " << rss << std::endl;

	for (size_t i = 0; i < population.size(); i++) {
		Node * p = population[i];

		dominated_solutions[p].reserve(population.size() / 2);
		domination_counts[p] = 0;

		for(size_t j = 0; j < population.size(); j++) {
			if (i == j)
				continue;
			Node * q = population[j];

			if(p->Dominates(q))
				dominated_solutions[p].push_back(q);
			else if (q->Dominates(p))
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
	}

	std::cout<<"number of layers:"<<nondominated_fronts.size()<<std::endl;
	process_mem_usage(vm, rss); std::cout << "At the start of FastNonDominatedSorting_adaptive_alpha_dominance-flag4 VM: " << vm << "; RSS: " << rss << std::endl;



	for(auto it = dominated_solutions.begin(); it != dominated_solutions.end();it++)
	{
		it->second.clear();
		it->second.shrink_to_fit();
	}

	dominated_solutions.clear();//dominated_solutions.rehash(0);
	domination_counts.clear();//domination_counts.rehash(0);

	//std::map<Node*, std::vector<Node*>>().swap(dominated_solutions);
	//std::map<Node*,int>().swap(domination_counts);

	domination_counts.rehash(0);
	dominated_solutions.rehash(0);

	current_front.clear();
	current_front.shrink_to_fit();



	return nondominated_fronts;
}


void NSGA2GenerationHandler::ComputeCrowndingDistance(std::vector<Node *> & front){
	size_t number_of_objs = front[0]->cached_objectives.size();
	size_t front_size = front.size();

	for (Node * p : front) {
		p->crowding_distance = 0;
	}

	for(size_t i = 0; i < number_of_objs; i++) {
		std::sort(front.begin(), front.end(), [i](const Node * lhs, const Node * rhs)
			{
			    return lhs->cached_objectives[i] < rhs->cached_objectives[i];
			});

		front[0]->crowding_distance = datum::inf;
		front[front.size() - 1]->crowding_distance = datum::inf;

		double_t min_obj = front[0]->cached_objectives[i];
		double_t max_obj = front[front.size() - 1]->cached_objectives[i];

		if (min_obj == max_obj)
			continue;

		for(size_t j = 1; j < front.size() - 1; j++) {
			if (isinf(front[j]->crowding_distance))
				continue;

			double_t prev_obj, next_obj;
			prev_obj = front[j-1]->cached_objectives[i];
			next_obj = front[j+1]->cached_objectives[i];

			front[j]->crowding_distance += (next_obj - prev_obj)/(max_obj - min_obj);
		}
	}
}




bool comp_NDSORT(Node* n1, Node* n2)
{
	if((n1->cached_objectives[0] < n2->cached_objectives[0]) || (n1->cached_objectives[0] == n2->cached_objectives[0] && n1->cached_objectives[1] < n1->cached_objectives[1]))
	{
		return true;
	}
	else
		return false;
}


//improved NDSORT: based on the following paper:
//reducing the run-time coimplexity of multiobjective EAs: The NSGA-II and other algorithms
std::vector<std::vector<Node*>> NSGA2GenerationHandler::FastNonDominatedSorting_IMPROVEDHEFFICIENCY(std::vector<Node*> & population){
	
	std::vector<std::vector<Node *>> fronts;
	fronts.reserve(int(population.size()*0.05));

	std::vector<Node *> population_sorted;
	population_sorted.assign(population.begin(),population.end());
	sort(population_sorted.begin(),population_sorted.end(),comp_NDSORT);
	
	fronts.push_back(std::vector<Node *>());
	fronts[0].push_back(population_sorted[0]);

	for(int i = 1; i < population_sorted.size(); i++){
		bool dominated = false;
		//determine if the current individual is dominated by elements in the current last front
		int last_front_seq = fronts.size() - 1;
		int last_ind_Seq = fronts[last_front_seq].size()-1;
		if(fronts[last_front_seq][last_ind_Seq]->cached_objectives[1] < population_sorted[i]->cached_objectives[1]){
			// current solution is dominated by the last layer, and we need to append a new layer followed by the current last layer
			fronts.push_back(std::vector<Node *>());
			fronts[fronts.size()-1].push_back(population_sorted[i]);
			continue;
		}else{
			//current solution is not dominated by the last layer, then we should use bisection search to find the last that this individual belongs to
			std::vector<std::vector<Node*>>::iterator it_head = fronts.begin();
			std::vector<std::vector<Node*>>::iterator it_tail = fronts.end();
			while(it_head != it_tail)
			{
				std::vector<std::vector<Node*>>::iterator it_middle = it_head + int((it_tail - it_head)/2);
				if((*it_middle)[it_middle->size()-1]->cached_objectives[1] < population_sorted[i]->cached_objectives[1]){
					it_head = it_middle;
				}else{
					it_tail = it_middle;
				}
			}
			it_head->push_back(population_sorted[i]);
		}
	}

	return fronts;
}