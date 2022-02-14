

#include "GPGOMEA/Evolution/NSGA2DPGenerationHandler.h"
using namespace std;
using namespace arma;

void NSGA2DPGenerationHandler::PerformGeneration(std::vector<Node *> & population)
{
	std::cout<<"PerformGeneration_Non_duplicate"<<std::endl;

	// Keep track of normal pop_size
	size_t pop_size = population.size();

	// Parent selection based on MO tournaments
	vector<Node*> selected_parents; selected_parents.reserve(population.size());
	while(selected_parents.size() < pop_size) {
		selected_parents.push_back(TournamentSelection::GetMOTournamentSelectionWinner(population, conf->tournament_selection_size));
	}

    // Variation
    std::vector<Node*> offspring = MakeOffspring(population, selected_parents);

    // Update fitness
    fitness->GetPopulationFitness(offspring, true, conf->caching);

    // P+O
    population.insert(population.end(), offspring.begin(), offspring.end());

	//put the duplicated solutions to the last layer
	//std::set<std::vector<double>,cmpKey_set_vectorDOUBLE> uniq_filter;
	std::set<std::vector<double>> uniq_filter;
	std::vector<Node *> uniq_popu;
	std::vector<Node *> last_layer;
	for(int i = 0; i < population.size(); i++)
	{
		if(uniq_filter.find(population[i]->semantic_description)!= uniq_filter.end()){
			last_layer.push_back(population[i]);
		}
		else{
			uniq_popu.push_back(population[i]);
			uniq_filter.insert(population[i]->semantic_description);
		}
	}
	for(int i = 0; i < last_layer.size(); i++)
	{
		last_layer[i]->rank = 999999;
	}

    // Assign ranks
	std::vector<std::vector<Node*>> fronts;
	fronts = FastNonDominatedSorting(uniq_popu);
	fronts.push_back(last_layer);
	//fronts = FastNonDominatedSorting(population);

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