#include "GPGOMEA/Evolution/LengthControlTruncationGenerationHandler.h"

#include <malloc.h>

using namespace std;
using namespace arma;

template<typename T> void display_vector(std::vector<T> vec, string str)
{

    std::cout<<std::endl<<str<<std::endl;
    for(int i = 0; i < vec.size(); i++)
        std::cout<<vec[i]<<" ";
    std::cout<<std::endl<<std::endl;
}

template<typename T> T sum_vector(std::vector<T> vec)
{
	T sum = 0;
	for(int i = 0; i < vec.size(); i++)
		sum += vec[i];
	return sum;
}

void show_population(vector<Node *> popu, string str)
{
	std::cout<<std::endl<<str<<std::endl;

	for(int i = 0; i < popu.size(); i++){
		std::cout<<"ind "<<i<<": length="<<popu[i]->cached_objectives[0]<<", MSE="<<popu[i]->cached_objectives[1]<<", EXP:"<<popu[i]->GetSubtreeExpression()<<std::endl;
	}
	std::cout<<endl;
}

template<typename T> T get_max_vector(std::vector<T> vec)
{
	if(vec.size() == 0)
		return -1;
	T max_val = vec[0];
	for(int i = 1; i < vec.size(); i++)
	{
		if(vec[i] > max_val)
			max_val = vec[i];
	}

	return max_val;
}

template<typename T1, typename T2> void show_map(map<T1, T2> m, size_t max_len, string str)
{
	std::cout<<std::endl<<str<<std::endl;
	for(int i = 1; i < max_len; i++)
	{
		if(m.find(i)!= m.end())
		{
			cout<<i<<" "<<m[i]<<std::endl;
		}
		else
		{
			cout<<i<<" "<<0<<std::endl;
		}
	}
	std::cout<<std::endl;
	
}


double LengthControlTruncationGenerationHandler::calculate_middle_accuracy(std::vector<Node *> & population)
{
	std::vector<double> accuracy_list;
	for(int i = 0; i < population.size(); i++)
	{
		accuracy_list.push_back(population[i]->cached_fitness);
	}
	std::sort(accuracy_list.begin(), accuracy_list.end());
	int cut_point = -1;
	double middle_fitness = 99999;
	if( population.size()%2==0)
	{
		cut_point = int(population.size()/2);
		middle_fitness = (accuracy_list[cut_point] + accuracy_list[(cut_point+1)])/2;
	}
	return middle_fitness;
}



void LengthControlTruncationGenerationHandler::PerformGeneration(std::vector<Node*> & population) {
    
    size_t pop_size = population.size();

	std::vector<std::vector<Node *>> popu_front = FastNonDominatedSorting(population);

	for(int i = 0; i < popu_front.size(); i++)
	{
		ComputeCrowndingDistance(popu_front[i]);
	}

    //reserve data structure for Variation
	std::vector<Node*> offspring;
	
	std::cout<<"cleaning data structure...";
	crossover_attempts.assign(conf->maximum_solution_size,0);
    crossover_succeed.assign(conf->maximum_solution_size,0);
    crossover_probability.assign(conf->maximum_solution_size,0);
    mutation_attempts.assign(conf->maximum_solution_size,0);
    mutation_succeed.assign(conf->maximum_solution_size,0);
    mutation_probability.assign(conf->maximum_solution_size,0);


	std::cout<<"tournament selection...";
    vector<Node*> selected_parents; selected_parents.reserve(population.size());
    while(selected_parents.size() < pop_size){
	    selected_parents.push_back(TournamentSelection::GetMOTournamentSelectionWinner(population, conf->tournament_selection_size));
	}

	//calculate middle value of accuracy
	double middle_accuracy = calculate_middle_accuracy(population);

	//variation
	std::cout<<"MakeOffspring (variation)...";
	offspring = MakeOffspring_counting_improvment_frequency_with_LCTrunction(population, selected_parents, crossover_attempts, mutation_attempts);
		
    // Update fitness
	std::cout<<"update fitness...";
    fitness->GetPopulationFitness(offspring, true, conf->caching);

	//update probibality matrix
	std::cout<<"update probibality...";
	use_interpol = true;
	//use_interpol = false;
    update_probability(offspring, population, middle_accuracy, use_interpol);
	//update_probability_linear_interpolation(offspring, population, middle_accuracy);
    
	// P+O
	std::cout<<"P+O...";
    population.insert(population.end(), offspring.begin(), offspring.end());

	// Pick survivors
	std::cout<<"Pick survivors...";
    std::vector<Node*> selection = survivors_selection(population, pop_size);
	
	//for(int i = 0; i < selection.size(); i++)
	//	std::cout<<i<<":"<<selection[i]->GetSubtreeExpression()<<" MSE="<<selection[i]->cached_fitness<<std::endl;
	

	for (Node * n : population) 
	{
		n->ClearSubtree();
	}
	population.clear();
	population = selection;

	std::cout<<"pop size after selection:"<<population.size()<<std::endl;
    
	// Assign ranks
	//std::vector<std::vector<Node*>> fronts = FastNonDominatedSorting(population);

	//update archive
	vector<Node *> new_archive;
	for(int i = 0; i < population.size(); i++)
	{
		if (population[i]->rank == 0)
			new_archive.push_back(population[i]);
	}
	for(int j = 0; j < archive.size(); j++)
	{
		new_archive.push_back(archive[j]);
	}
	vector<vector<Node *>> fronts_new_archive = FastNonDominatedSorting(new_archive);
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

	//FastNonDominatedSorting(population);

	save_population(population);

	malloc_trim(0);

}


std::vector<Node *> LengthControlTruncationGenerationHandler::survivors_selection(std::vector<Node *> & population, size_t survivor_size){
	
	std::vector<std::vector<Node*>> fronts = FastNonDominatedSorting(population);
    std::vector<Node*> selection; selection.reserve(survivor_size);

	std::map<size_t,size_t> copy_of_limits = size_limits_for_selection; // this should create a map that is a copy of the other map.
	for(std::map<size_t,size_t>::iterator it = copy_of_limits.begin(); it != copy_of_limits.end(); it++) {
		std::cout << "\tsize " << it->first << " ; limit " << it->second << endl;
	}

	size_t tot_limits_remaining = std::accumulate(std::begin(copy_of_limits), std::end(copy_of_limits), 0,
				[](const std::size_t previous, const auto& element){ return previous + element.second; });
	//bool parsed_once = false;
	std::vector<bool> CD_calculated(fronts.size(), false);
	while(selection.size() < survivor_size) {

		size_t current_front_idx = 0;
		
		while ( current_front_idx < fronts.size() && selection.size() < survivor_size ) {
			
			//if (!parsed_once) { // compute CD only the first time we are selecting
			if(CD_calculated[current_front_idx] == false)
			{
				ComputeCrowndingDistance(fronts[current_front_idx]);
				sort( fronts[current_front_idx].begin(), fronts[current_front_idx].end(), [](const Node * lhs, const Node * rhs) 
				{
					return lhs->crowding_distance > rhs->crowding_distance;
				});
				CD_calculated[current_front_idx] = true;
			}
			
			//selection.insert(selection.end(), fronts[current_front_idx].begin(), fronts[current_front_idx].end());
			for( Node * solution_in_front : fronts[current_front_idx]) {
				// copy the solution if within the limit for its size
				size_t size_of_solution = solution_in_front->GetSubtreeNodes(true).size();

				if (copy_of_limits[size_of_solution] > 0) {
					selection.push_back(solution_in_front->CloneSubtree());
					copy_of_limits[size_of_solution]--;
					//string bla; cin >> bla;
					// check that we can still copy stuff (i.e., there exist sizes for which the limit is > 0)
					size_t tot_limits_remaining = std::accumulate(std::begin(copy_of_limits), std::end(copy_of_limits), 0,
                                          [](const std::size_t previous, const auto& element){ return previous + element.second; });
					//cout << "tot_limits_remaining is " << tot_limits_remaining << endl;
					if (tot_limits_remaining == 0) {
						// then regenate the limits & proceed
						copy_of_limits = size_limits_for_selection;
					}
		
				}

				// check if we should stop
				if (selection.size() == survivor_size) {
					break;
				}
			}
			if (selection.size() == survivor_size) {
				break;
			}
			current_front_idx++;
		}

		//parsed_once = true;
	}

	return selection;

}


//population is P+O, selection is empty with reserved memory space. But NEVER using index to store element in selection, push_back should still be used
std::vector<Node *> LengthControlTruncationGenerationHandler::survivors_selection_old(std::vector<Node *> & population, std::vector<Node *> & selection)
{
	selection.clear();
	std::cout<<std::endl<<"goto survivors_selection...";
    std::vector<int> quantity_individuals_ExpLen_in_population_runtime(quantity_individuals_ExpLen_in_population);
    
	std::vector<std::vector<Node*>> fronts = FastNonDominatedSorting(population);
    for(int i = 0; i < fronts.size(); i++)
    {
        //search for solutions of different expression length layer by layer
        for(int j = 0; j < fronts[i].size(); j++)
		{
			int len = fronts[i][j]->GetSubtreeNodes(true).size();
			if(quantity_individuals_ExpLen_in_population_runtime[len-1] >= 1)
			{
				selection.push_back(fronts[i][j]->CloneSubtree());
				//std::cout<<"len="<<len<<", insert a solution!"<<std::endl;
				quantity_individuals_ExpLen_in_population_runtime[len-1] -= 1;
			}
		}
    }

	for(int i = 0; i < quantity_individuals_ExpLen_in_population_runtime.size(); i++)
	{
		//if after traverse all solutions in different layer and there is still some exp length need solution, there are several situations:
		//1). if there is solutions in the population, then insert them(repeatedly) 
		//2). if there is no solutions in the population, then pick up from repo
		//3). if there is no solutions in repo, then randomly pick up solutions from the first layer
		
		//following is the step of 1)
		//if the number of individuals in the first layer >= the number of individuals need to be inserted, then select individuals from the first layer
		//other wise, randomly pick up individuals from all of the layers
		
		
		if(quantity_individuals_ExpLen_in_population_runtime[i] > 0)
		{
			//std::cout<<"	still need to add "<<quantity_individuals_ExpLen_in_population_runtime[i]<<" solutions"<<std::endl;
			int sum_whole_popu = 0;
			std::vector<std::vector<Node *>> fronts_filtered_expLength;
			for(int j = 0; j < fronts.size(); j++)
			{
				std::vector<Node *> vec;
				fronts_filtered_expLength.push_back(vec);
			}
			for(int j = 0; j < fronts.size(); j++)
			{
				for(int k = 0; k < fronts[j].size(); k++)
				{
					if(fronts[j][k]->GetSubtreeNodes(true).size() == (i+1)) //i=0 correspondent to the expression length of 1, X+1 correspondent to the expression length of X
					{
						sum_whole_popu++;
						fronts_filtered_expLength[j].push_back(fronts[j][k]);
					}
				}
			}

			if(sum_whole_popu != 0)
			{
				bool finished_insert = false;
				while(quantity_individuals_ExpLen_in_population_runtime[i] > 0)
				{
					for(int j = 0; j < fronts_filtered_expLength.size(); j++)
					{
						for(int k = 0; k < fronts_filtered_expLength[j].size(); k++)
						{
							selection.push_back(fronts_filtered_expLength[j][k]->CloneSubtree());
							quantity_individuals_ExpLen_in_population_runtime[i] -= 1;
							if(quantity_individuals_ExpLen_in_population_runtime[i] == 0)
							{
								finished_insert = true;
								break;
							}
							else if(quantity_individuals_ExpLen_in_population_runtime[i] < 0)
							{
								throw std::runtime_error("error quantity_individuals_ExpLen_in_population_runtime[i]<0");
								
							}
						}
						if(finished_insert == true)
							break;
					}
					if(finished_insert == true)
						break;
				}
				
			}
			else if(sum_whole_popu == 0)
			{
				bool filled = false;
				
				for(int len = conf->maximum_solution_size; len > 0; len--)
				{
					if(len % 2 == 0)
						continue;
					for(int j = 0; j < fronts.size(); j++)
					{
						for(int k = 0; k < fronts[j].size(); k++)
						{
							if(fronts[j][k]->GetSubtreeNodes(true).size() == len)
							{
								selection.push_back(fronts[j][k]->CloneSubtree());
								quantity_individuals_ExpLen_in_population_runtime[i] -= 1;
								if(quantity_individuals_ExpLen_in_population_runtime[i] == 0)
								{
									filled = true;
									break;
								}
								else if(quantity_individuals_ExpLen_in_population_runtime[i] < 0)
								{
									throw std::runtime_error("error quantity_individuals_ExpLen_in_population_runtime[i]<0");
								}
							}
						}
						if(filled == true)
							break;
					}
					if(filled == true)
						break;
				}
			}
		}
		else if(quantity_individuals_ExpLen_in_population_runtime[i] < 0)
		{
			throw std::runtime_error("error quantity_individuals_ExpLen_in_population_runtime[i] < 0");
		}
	}

	for(int i = 0; i < quantity_individuals_ExpLen_in_population_runtime.size(); i++)
	{
		if(quantity_individuals_ExpLen_in_population_runtime[i] != 0)
		{
			std::cout<<"quantity_individuals_ExpLen_in_population_runtime value is not zero, error!"<<std::endl;
			std::cout<<"trace data:"<<std::endl;
			display_vector(quantity_individuals_ExpLen_in_population_runtime,"quantity_individuals_ExpLen_in_population_runtime:");
			exit(-1);
		}
	}

	return selection;
}



//if middle_accuracy = -1, then determine if offspring is great by comparing them with parents, if middle_accuracy >= 0 then compare offspring with the middle accuracy of the parent population to determine if offspring is good
void LengthControlTruncationGenerationHandler::update_probability(const vector<Node *> & offspring, const vector<Node *> & population, double middle_accuracy, bool use_interpol)
{
	size_limits_for_selection.clear();
    std::cout<<"Update_probability...start to refresh succeed table...";

    std::cout<<"middle accuracy="<<middle_accuracy<<std::endl;

	// check what are all the possible sizes
	set<size_t> all_current_sizes;
	for(Node * o : offspring) {
		size_t size_offspring = o->GetSubtreeNodes(true).size();
		size_t size_parent = o->parent_exp_len;
		all_current_sizes.insert(size_offspring);
		all_current_sizes.insert(size_parent);
	}

	// now store all success rates
	map<size_t,double_t> sizes_n_attempts;
	map<size_t,double_t> sizes_n_successes;
	map<size_t,double_t> sizes_n_success_rates;
	

	//update(or refresh) succeed table
    for(int i = 0; i < offspring.size(); i++)
    {
		// update attempts
		sizes_n_attempts[offspring[i]->parent_exp_len] = sizes_n_attempts[offspring[i]->parent_exp_len] + 1;

		if(middle_accuracy == -1)
		{
			// TODO: this is not being used, should be updated if one wants to use it
			//throw std::runtime_error("This code is not ready yet!");
			if(offspring[i]->cached_fitness < offspring[i]->parent_fitness)
				sizes_n_successes[offspring[i]->parent_exp_len] = sizes_n_successes[offspring[i]->parent_exp_len] + 1;
		}
		else if(middle_accuracy >= 0)
		{
			if(offspring[i]->cached_fitness < middle_accuracy)
				sizes_n_successes[offspring[i]->parent_exp_len] = sizes_n_successes[offspring[i]->parent_exp_len] + 1;
		}
		else
		{
			throw std::runtime_error("error, middle_accuracy is not set to >= 0 or == -1, cannot determine how to evaluate if offspring is in good quality");
		}
        
    }

	//display_vector(crossover_succeed,"IN UPDATEPROBIBALITY FUNCTION, crossover_succeed:");
	//display_vector(crossover_attempts,"IN UPDATEPROBIBALITY FUNCTION, crossover_attempts:");
	//display_vector(mutation_succeed,"IN UPDATEPROBIBALITY FUNCTION, mutation_succeed:");	
	//display_vector(mutation_attempts,"IN UPDATEPROBIBALITY FUNCTION, mutation_attempts:");


	// Now compute sizes_n_success_rates （& compute stuff to normalize along the way)
	double_t total_success_rate = 0;
	for(size_t size : all_current_sizes) {
		if (sizes_n_attempts.find(size) != sizes_n_attempts.end()){ // TODO: check this syntax is correct (for checking if something exists in a map)
			sizes_n_success_rates[size] = ((double_t)sizes_n_successes[size]) / ((double_t)sizes_n_attempts[size]);
			total_success_rate += sizes_n_success_rates[size];
		} else {
			sizes_n_success_rates[size] = 0.0;
		}
	}
	
	show_map(sizes_n_success_rates,conf->maximum_solution_size,"before or without Interpolation, the sizes_n_success_rates is:");

	// TODO: if we are interpolating, we need to check if the reason why this is 0 is because #attempts for this size is 0;
	// in that case, instead of setting it to 1, we set it to the interpolated value of the number of successes
	bool interpolation = use_interpol;
	if (interpolation) {
		std::cout<<"NOW INTERPOLATION HAS BEEN USED"<<std::endl;
		vector<pair<size_t,double_t>> interpol_quantity;
		for(size_t size: all_current_sizes) {
			if (sizes_n_attempts[size] != 0) {
				continue;
			} else {
				// we need to interpolate here by finding the closest sizes size' and size'' for which sizes_n_attempts[size'] != 0; 
				// and then take their interpolated success rate.
				int exp_len_left = -1;  double_t dist_left = arma::datum::inf;
				int exp_len_right = -1; double_t dist_right = arma::datum::inf;
				double_t s_left = 0;  double_t quantity_left = 0;
				double_t s_right = 0; double_t quantity_right = 0;

				for(size_t exp_size_now: all_current_sizes){
					if(exp_size_now < size && sizes_n_attempts[exp_size_now] != 0 && (size-exp_size_now) < dist_left){
						exp_len_left = exp_size_now;
						dist_left = size - exp_size_now;
					}
					else if(exp_size_now > size && sizes_n_attempts[exp_size_now] != 0 && (exp_size_now - size) < dist_right){
						exp_len_right = exp_size_now;
						dist_right = exp_size_now - size;
					}
				}
				
				if((exp_len_left < 1  || exp_len_left > conf->maximum_solution_size) && (exp_len_right < 1  || exp_len_right > conf->maximum_solution_size))//if left side and right side are all zero, then also put zero to this expression length (this situation could never happened, I think this is a kind of error)
					throw std::runtime_error("during interpolation, the expression length at both sides of the current expression length can't be detected.");
				if((exp_len_left > 0 && exp_len_left >= size) || (exp_len_right > 0 && exp_len_right <= size))
					throw std::runtime_error("during interpolation, the expression_length_left or expression_len_right excess the boundary");
				s_left = 0; quantity_left = 0;
				s_right = 0; quantity_right = 0;
				if(exp_len_left >= 1 && exp_len_left <= conf->maximum_solution_size)
					quantity_left = sizes_n_success_rates[exp_len_left];
				if(exp_len_right >= 1 && exp_len_right <= conf->maximum_solution_size)
					quantity_right = sizes_n_success_rates[exp_len_right];
				
				double_t interpol_success_rate;
				if(quantity_left == 0 && quantity_right != 0)
					interpol_success_rate = quantity_right;
				else if(quantity_right == 0 && quantity_left != 0)
					interpol_success_rate = quantity_left;
				else{
					s_left = 1-(((double)size - exp_len_left)/((double)exp_len_right - exp_len_left));
					s_right = 1 - (((double)exp_len_right-size)/((double)exp_len_right - exp_len_left));
					interpol_success_rate = quantity_left*s_left + quantity_right*s_right;
				}
				
				
				//the interpole value should be between the probibality of left side and right side, and should never lower than the probibality of left side of larger than the right side
				double_t quantity_small = quantity_left < quantity_right ? quantity_left : quantity_right;
				double_t quantity_large = quantity_left > quantity_right ? quantity_left : quantity_right;
				
				if(fabs(interpol_success_rate-quantity_small) < 0.001 && fabs(interpol_success_rate-quantity_large) < 0.001){
					std::cout<<"interpol_success_rate="<<interpol_success_rate<<", quantity_small="<<quantity_small<<", quantity_large="<<quantity_large<<", they are the same, the interpol value should also the same"<<std::endl;
				}
				if(interpol_success_rate < quantity_small && fabs(interpol_success_rate - quantity_small ) > 0.001){
					std::cout<<"interpol_success_rate - quantity_small = "<<interpol_success_rate - quantity_small<<", interpol_success_rate - quantity_large = "<<interpol_success_rate - quantity_large<<std::endl;
					string err_str = "interpol_success_rate error, quantity left="+to_string(quantity_left)+", interpol value:"+to_string(interpol_success_rate)+", quantity right="+to_string(quantity_right)+",explen left = "+to_string(exp_len_left)+", explen right ="+to_string(exp_len_right)+", size now="+to_string(size)+",s_left="+to_string(s_left)+",s_right="+to_string(s_right)+", quantity_small="+to_string(quantity_small)+",quantity_large="+to_string(quantity_large);
					std::cout<<err_str<<std::endl;
					exit(-1);
					//throw std::runtime_error(err_str);
				}
				else if(interpol_success_rate > quantity_large && fabs(interpol_success_rate - quantity_large) > 0.001){
					std::cout<<"interpol_success_rate - quantity_small = "<<interpol_success_rate - quantity_small<<", interpol_success_rate - quantity_large = "<<interpol_success_rate - quantity_large<<std::endl;
					string err_str = "interpol_success_rate error, quantity left="+to_string(quantity_left)+", interpol value:"+to_string(interpol_success_rate)+", quantity right="+to_string(quantity_right)+",explen left = "+to_string(exp_len_left)+", explen right ="+to_string(exp_len_right)+", size now="+to_string(size)+",s_left="+to_string(s_left)+",s_right="+to_string(s_right)+", quantity_small="+to_string(quantity_small)+",quantity_large="+to_string(quantity_large);
					std::cout<<err_str<<std::endl;
					exit(-1);
					//throw std::runtime_error(err_str);
				}
					
				if(interpol_success_rate != 0){
					pair<size_t,double_t> p;
					p.first = size;
					p.second = interpol_success_rate;
					interpol_quantity.push_back(p);
				}
			}
		}
		for(auto p : interpol_quantity)
		{
			if(sizes_n_success_rates[p.first] != 0)
				throw std::runtime_error("error, trying to interpol a probibality to expression length that already has probibality");
			sizes_n_success_rates[p.first] = p.second;
		}
		show_map(sizes_n_success_rates,conf->maximum_solution_size,"after Interpolation, the sizes_n_success_rates is:");

		//if we use interpolation to add some probibality to some expressions, then we need to update the sum_up of success rate (for normalization)
		total_success_rate = 0;
		for(auto p: sizes_n_success_rates)
			total_success_rate += p.second;

		//try to normalize and check if the number of individuals assigned by interpolation is too large (than the number of individuals that exist in the population in reality)
		/*
		for(size_t size : all_current_sizes) {
			size_limits_for_selection[size] = round( sizes_n_success_rates[size] / ((double_t)total_success_rate)* population.size() );	
		}
		for(auto p: interpol_quantity)
		{

			for(auto q: population)
			{
				if()
			}
		}
		*/
	}
	//finish interpol operation


	

	// normalize everything （w.r.t. pop size)
	for(size_t size : all_current_sizes) {
		size_limits_for_selection[size] = round( sizes_n_success_rates[size] / ((double_t)total_success_rate)* population.size() );
		if  (sizes_n_success_rates[size] == 0){
			// force the limit to be at least 1
			size_limits_for_selection[size] = 1;		
		} else if (sizes_n_success_rates[size] > sizes_n_attempts[size]) {
			std::cout<<"sizes_n_success_rates[size] > sizes_n_attempts[size], enforce the number to be the same as attempts!"<<std::endl;
			sizes_n_success_rates[size] = sizes_n_attempts[size]; // TODO: can we do something better?
		}
	}


	return;
	
}






std::vector<std::vector<Node*>> LengthControlTruncationGenerationHandler::FastNonDominatedSorting(const std::vector<Node*> & population){
	size_t rank_counter = 0;
	vector<vector<Node*>> nondominated_fronts; nondominated_fronts.reserve(10);
	unordered_map<Node*, std::vector<Node*>> dominated_solutions;
	unordered_map<Node*, int> domination_counts;
	vector<Node*> current_front; current_front.reserve(population.size() / 2);


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

	return nondominated_fronts;
}


void LengthControlTruncationGenerationHandler::ComputeCrowndingDistance(std::vector<Node *> & front){
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


double LengthControlTruncationGenerationHandler::calculate_distance(Node*  n1, Node* n2)
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


void LengthControlTruncationGenerationHandler::save_population(std::vector<Node*> & population)//final population is the combination of archive and current population(and only pick up non-dominated solutions)
{
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


	char str_show[512];
	sprintf(str_show,"finish iteration of run:%d/%d, the population:",conf->current_generation,conf->max_generations);
	show_population(population,str_show);


	if(conf->current_generation == conf->max_generations)
	{
		save_final_population(population);	
		for (Node * n : archive){
			n->ClearSubtree();//DO NOT FORGET TO CLEAN ARCHIVE WHEN ITERATION FINISHED
		}
	}

}

void LengthControlTruncationGenerationHandler::save_final_population(std::vector<Node*> & population)//final population is the combination of archive and current population(and only pick up non-dominated solutions)
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
	
    //sprintf(savefilename, "./PF/PF_%s_runs%d.txt","LengthControl",conf->num_of_run);
    //sprintf(savefilename_archive, "./PF/PF_%s_archive_runs%d.txt","LengthControl",conf->num_of_run);

	if(use_interpol == true){
		sprintf(savefilename, "./PF/PF_%s_runs%d.txt","LengthControl-interpol",conf->num_of_run);
    	sprintf(savefilename_archive, "./PF/PF_%s_archive_runs%d.txt","LengthControl-interpol",conf->num_of_run);
	}
	else{
		sprintf(savefilename, "./PF/PF_%s_runs%d.txt","LengthControl",conf->num_of_run);
    	sprintf(savefilename_archive, "./PF/PF_%s_archive_runs%d.txt","LengthControl",conf->num_of_run);
	}
	

    //save pf of the last generation
    std::vector<std::vector<Node *>> fronts = FastNonDominatedSorting(population);
	std::fstream fout;
	fout.open(savefilename, std::ios::out);
	for(int i = 0; i < fronts[0].size(); i++)
	{
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
    std::vector<std::vector<Node *>> fronts_archive = FastNonDominatedSorting(combined_popu);
	fout.open(savefilename_archive, std::ios::out);
	for(int i = 0; i < fronts_archive[0].size(); i++)
	{
		fout<<fronts_archive[0][i]->cached_objectives[0]<<" "<<fronts_archive[0][i]->cached_objectives[1]<<"\r\n";
	}
	fout.close();
}





void LengthControlTruncationGenerationHandler::update_archive(std::vector<Node *> & population)
{
	vector<vector<Node *>> layer_population = FastNonDominatedSorting(population);
	for(int i = 0; i < layer_population[0].size(); i++)
	{
		Node * p = layer_population[0][i]->CloneSubtree();
		archive.push_back(p);
	}

	vector<vector<Node *>> layer_archive = FastNonDominatedSorting(archive);
	archive = layer_archive[0];

	for(int i = 1; i < layer_archive.size(); i++)//remain the 1st layer to be new archive, while the remaining elements are removed from the archive and deleted from the memory
	{
		for (Node * n : layer_archive[i]) 
		{
			n->ClearSubtree();
		}
	}

	//there is high possibility that there are repeated solutions in the population and thus we should filter the archive to maintain unique elite individuals in the archive
	std::set<std::pair<double,double>,compare_pair_double_double_for_set> p_unique;
	std::vector<Node *> archive_filter;
	for(int i = 0; i < archive.size(); i++)
	{
		pair<double,double> p;
		p.first = archive[i]->cached_objectives[0];
		p.second = archive[i]->cached_objectives[1];
		if(p_unique.find(p) == p_unique.end())
		{
			archive_filter.push_back(archive[i]);
			p_unique.insert(p);
		}
		else
		{
			archive[i]->ClearSubtree();
		}
	}
	archive.clear();
	archive = archive_filter;

}