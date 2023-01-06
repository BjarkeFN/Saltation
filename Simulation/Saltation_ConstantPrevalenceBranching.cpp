#include <iostream>
#include <fstream>
#include <armadillo>
#include <experimental/random>
#include <algorithm>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <ctime>
#include <cmath>
#include <bitset>
#include <stack>
#include <unistd.h> // Unix-specific, for sleep() etc

using namespace std;
using namespace arma;

struct Params {
	int generations; // Number of generations to simulate
	float R0_initial; // R0 in the first generation (for all strains)
	float alpha; // Dispersion parameter, also known as k.
	int I_initial; // Initially infected persons.
	int N_L; // Length of 'genome'
	float k_mut; // Dispersion parameter of mutation rate distribution
	vector<float> popsizes; // Relative sizes of metapopulations (sum 1)
	mat popsizes_t; // Population sizes for all generations
	mat interpop; // Interaction matrix for metapopulations
	map<int,vector<int>> popmembers; // Membership matrix for metapopulations, M[i,:] = members of population i.
	vector<int> agentpop;  // Same information, but indexed such that M[i] = j => agent i is member of population.
	int N_epitopes; // Number of 'epitopal regions' (separate regions of the genome)
	int N_H;	// Number of high fitness configurations within each epitope
	int L_epitope;	// Length of epitopes. Must be such that N >= N_epitopes*L_epitope;
					// Note that this leaves room to leave some regions of the genome wholly neutral
	int d; // Distance to fit configuration, within which configurations are deleterious
	umat fitconfigs; // Shape (N_epitopes, N_H). Each fit config is denoted by its decimal representation
	mat fitconfigs_fitness; // Shape (N_epitopes, N_H) Actual fitness associated with each fit configuration.
	map<int, map<int, float>> unfit_configs_fitness;	// The unfit configurations are determined on-the-fly
														// i.e. once a new configuration arises by mutation
	float dR_H_mean; // Mean fitness contribution of a fit variant
	float dR_L_mean; // Mean fitness contribution of an unfit variant (should presumably be negative!) 
	float mut_rate_high; // Typical size of saltations
};

struct gtype_R {
	vector<int> gtype;
	float R;
	int mutsize;
};

struct gtype_R_status {
	gtype_R gtypeR;
	int status;
};

struct dtype_status {
	int net;
	int del;
	int ben;
};

struct simdata {
	map<int, map<int, vector<int> > > genotypes;
	map<int, map<int, float> > R0s;
	vector<float> diversities;
	vector<int> generation_sizes;
	vector<int> trackedvar_incidences;
	map<int,mat> hamming_distributions_t;
	map<int,mat> hamming_origindists_t;
	mat status_dists_t;
	Params parameters;
};

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}


void print_intvec(std::vector<int> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
	cout << endl;
}

void print_floatvec(std::vector<float> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
	std::cout << endl;
}

void print_map_intvec(map<int, vector<int> > input) 
{
	for (int i = 0; i < input.size(); i++) {
			cout << "Element " << i << ": ";
			print_intvec(input[i]);
		}
}

void print_map_int(map<int, int > input) 
{
	for (int i = 0; i < input.size(); i++) {
			cout << "Element " << i << ": ";
			cout << input[i] << endl;
		}
}

void print_map_float(map<int, float > input) 
{
	for (int i = 0; i < input.size(); i++) {
			cout << "Element " << i << ": ";
			printf("%f\n", input[i]);
		}
}

string int_to_binary_string(int n)
{
    string r;
    while(n!=0) {r=(n%2==0 ?"0":"1")+r; n/=2;}
    return r;
}

vector<int> string_to_vector (string intstring) {
	int L = intstring.size();
	vector<int> outvector;
	for (char const &c: intstring) {
        outvector.push_back(c-'0');
    }
	return outvector;
}

vector<int> dec_to_bitvector(int n, int minlength) {
        vector<int> outvector;
        vector<int> outvector_prosp;

        string s = int_to_binary_string(n);
        outvector_prosp = string_to_vector(s);
        int lendiff = outvector_prosp.size()-minlength;
        if (lendiff < 0) {
                int absdiff = abs(lendiff);
                for (int i=0; i<absdiff; i++) {
                        // Pad with zeros
                        outvector.push_back(0);
                }
        }

        // Copy outvector_prosp into outvector:
        for (int i=0; i < outvector_prosp.size(); i++) {
                outvector.push_back(outvector_prosp[i]);
        }
        return outvector;
}


int choose_weighted(vec x, Col<float> w) { // variable values and weights
	float r = randu();
	float s = accu(w); // Sum of weights
	bool chosen = false;
	int i = x.n_elem-1;
	float c;
	while (!chosen) {
		if (i<0) {
			cout << "No element chosen ... resorting to last element." << endl;
			c = x(0);
			chosen = true;
			cout << "The information given was" << endl;
			cout << "Weights:" << endl;
			w.print();
			cout << "Values:" << endl;
			x.print();
			cout << "The random number was " << r << endl;
			cout << "and the last weight-sum was " << s << endl;
		}
		else {
			c = x(i);
			s = s - w(i);
			if (s < r) {
				chosen = true;
			}
		i--;
		}
	}
	return c;
}

int choose_weighted_rowvecs(rowvec x, rowvec w) { // variable values and weights
	float r = randu();
	float s = accu(w); // Sum of weights
	bool chosen = false;
	int i = x.n_elem-1;
	float c;
	while (!chosen) {
		if (i<0) {
			cout << "No element chosen ... resorting to last element." << endl;
			c = x(0);
			chosen = true;
			cout << "The information given was" << endl;
			cout << "Weights:" << endl;
			w.print();
			cout << "Values:" << endl;
			x.print();
			cout << "The random number was " << r << endl;
			cout << "and the last weight-sum was " << s << endl;
		}
		else {
			c = x(i);
			s = s - w(i);
			if (s < r) {
				chosen = true;
			}
		i--;
		}
	}
	return c;
}


// Function for returning strings of floats with higher precision
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// Function for hashing, to get a "more unique" seed.
// http://www.concentric.net/~Ttwang/tech/inthash.htm
unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

auto draw_single_infectivity(float p_inf, float alpha) {
	random_device rd;
    mt19937 gen(rd());
	exponential_distribution<> d(1.0/p_inf);
	float p;
	// Note: The mean of the gamma distribution is alpha * beta.
	// Since we want mean = p_inf, we set beta = p_inf/alpha.
	float beta = p_inf/alpha;
	gamma_distribution<> g(alpha, beta);
	p = g(gen);
	return p;
}

int idx_by_idx_hamming(vector<int> bitvec1, vector<int> bitvec2) {
	// Assumes that bitvecs are same length
	int L = bitvec1.size();
	int diff = 0;
	for (int i = 0; i < L; i++) {
		if (bitvec1[i] != bitvec2[i]) {
			diff ++;
		  }
	}
	return diff;
}

int bitvector_to_int(vector<int> bitvec) {
	int out=0;
	int L = bitvec.size();
	for (int i = 0; i < L; i++) {
		out = out + bitvec[i] * pow(2, (L - 1) - i);
	}
	return out;
}

vector<int> int_to_bitvector(int int_in) {
	vector<int> out_bitvector;
	
	
	return out_bitvector;
}

float average_intvec(std::vector<int> const& v){
    if(v.empty()){
        return 0.0;
    }

    auto const count = static_cast<float>(v.size());
    return ((float)reduce(v.begin(), v.end())) / count;
}

umat setup_fitconfigs(Params parameters) {
	int L = parameters.L_epitope;
	int N_epitopes = parameters.N_epitopes;
	int N_H = parameters.N_H; // High fit confs within each epitope
	umat fitconfigs(N_epitopes, N_H, fill::zeros);
	// Highly fit configurations can simply be denoted by their 
	// integer (decimal) representation. If the length of an epitope's
	// sequence is L (bits), the lowest such representation is of
	// course 0, while the largest is 2^L-1.
	// For practical reasons, we probably do not want to exceed L=30
	// or thereabouts.
	// Right now, max is 25 (see dec_to_bitvector calls below)
	int conf_max = (pow(2,L)-1);
	cout << "conf_max=" << conf_max << endl;
	int new_conf;
	bool distant_enough; 
	for (int i=0; i < N_epitopes; i++) {
		for (int j=0; j < N_H; j++) {
			// For each new configuration, we need to
			// check that it is not too close to an existing one
			distant_enough = false;
			new_conf = rand() % conf_max; // Generate prospective configuration
			vector<int> new_conf_bitvector = dec_to_bitvector(new_conf, 25);
			while (!distant_enough) {
				int prev_conf;
				int distance;
				distant_enough = true;
				// Routine to check if distance is sufficient
				// (of course, we only need to check the ones withing the same epitope)
				if (j==0) { // No other configurations within same epitope yet
					distant_enough = true;
				}
				else {
					for (int k=0; k<j; k++) { // Looping over the existing configurations
						prev_conf = fitconfigs(i,k);
						vector<int> prev_conf_bitvector = dec_to_bitvector(prev_conf, 25);
						// Get distance between prospective new_conf and prev_conf:
						int distance = idx_by_idx_hamming(new_conf_bitvector, prev_conf_bitvector);
						if (distance < parameters.d) {
							distant_enough = false;
						}
					}
				}
			}
			fitconfigs(i, j) = new_conf;
		}
	}
	return fitconfigs;
}

float setup_lowfitness_dR(Params parameters) {
	float dR;
	
	float dR_L_mean = parameters.dR_L_mean; // Average 'fitness contribution' dR
											// of low fitness configurations
	float k_dR = 10;
	float uniform_width=0.5;
	string dR_L_distribution = "delta"; // delta, uniform gamma
	
	// Setup gamma dist:
	random_device rd;
	mt19937 gen(rd());
	float beta = dR_L_mean/k_dR;
	gamma_distribution<> g(k_dR, beta);
	
	if (dR_L_distribution=="gamma") {
				dR = g(gen);
	}
	else if (dR_L_distribution=="uniform") {
		dR = (dR_L_mean-0.5*uniform_width) + uniform_width * randu();
	}
	else { // delta
		dR = dR_L_mean;
	}
	
	return dR;
}

mat setup_fitconfig_fitnesses(Params parameters) {
	int L = parameters.L_epitope;
	int N_epitopes = parameters.N_epitopes;
	int N_H = parameters.N_H; // High fit confs within each epitope
	float dR_H_mean = parameters.dR_H_mean; // Average 'fitness contribution' dR
											// of highly fit configurations
	float k_dR = 10;
	float uniform_width=0.5;
	string dR_H_distribution = "delta"; // delta, uniform gamma
	
	
	mat fitconfigs_fitness(N_epitopes, N_H, fill::zeros);
	
	// Setup gamma dist:
	random_device rd;
	mt19937 gen(rd());
	float beta = dR_H_mean/k_dR;
	gamma_distribution<> g(k_dR, beta);
		
	for (int i = 0; i < N_epitopes; i++) {
		for (int j=0; j < N_H; j++) {
			if (dR_H_distribution=="gamma") {
				fitconfigs_fitness(i,j) = g(gen);
			}
			else if (dR_H_distribution=="uniform") {
				fitconfigs_fitness(i,j) = (dR_H_mean-0.5*uniform_width) + uniform_width * randu();
			}
			else { // delta
				fitconfigs_fitness(i,j) = dR_H_mean;
			}
		}
	}
	return fitconfigs_fitness;
}


float genotype_status(vector<int> genotype, Params parameters) {
	int N = genotype.size();
	int L = parameters.L_epitope;
	int N_epitopes = parameters.N_epitopes;
	int N_H = parameters.N_H; // High fit confs within each epitope
	
	float fitness_contribution = 0;
	
	// First, go through the N_epitopes epitopes in the genotype:
	int start_index = 0;
	vector<int> subvector;
	for (int i = 0; i < N_epitopes; i++) {
		subvector.clear();
		for (int idx = start_index; idx < start_index+L; idx++) {
			subvector.push_back(genotype[idx]);
		}
		// Convert subvector to a (decimal) integer and check
		// if it is close to (or identical to) a fit configuration
		int conf = bitvector_to_int(subvector);
		bool conf_fit = false;
		bool conf_unfit = false;
		for (int c = 0; c < N_H; c++) {
			if (conf == parameters.fitconfigs(i,c)) {
				fitness_contribution = fitness_contribution + parameters.fitconfigs_fitness(i,c);
				conf_fit = true;
			}
			else {
				// Check if the configuration is within d of a fit one, and hence
				// subject to sign epistasis
				int distance;
				distance = idx_by_idx_hamming(subvector, dec_to_bitvector(parameters.fitconfigs(i,c), subvector.size()));
				if (distance < parameters.d) {
					conf_unfit = true;
					
					// Check if the fitness of this configuration has already been determined:
					if (parameters.unfit_configs_fitness[i].find(conf) != parameters.unfit_configs_fitness[i].end()) {
						// The fitness of this configuration has already been assigned
						fitness_contribution = fitness_contribution + parameters.unfit_configs_fitness[i][conf];
					}
					else { // Assign a fitness contribution value to this configuration
						float fit_contr_loc = setup_lowfitness_dR(parameters);
						parameters.unfit_configs_fitness[i][conf] = fit_contr_loc;
						fitness_contribution = fitness_contribution + parameters.unfit_configs_fitness[i][conf];
					}
				} 
			}
		}
		start_index = start_index + L;
	}
	
	//cout << "Fitness was: " << fitness << endl;
	return fitness_contribution;
}

gtype_R gen_mut(vector<int> genotype, float R0_initial, float mutation_rate, float dR, Params parameters) {
	// Reinterpret mutation rate as avg. number of flips
	// (as a Poissson process)
	random_device rd;
	mt19937 gen(rd());
	float R_min = 0.0;
	float R_max = 5;
	
	float R = R0_initial;
	//cout << "Initial R " << R << endl;
	gtype_R outdata;
	int number_of_flips;
	poisson_distribution<> g(mutation_rate);
	number_of_flips = g(gen);
	for (int f = 0; f < number_of_flips; f++) {
		int flip_idx = (rand() % genotype.size());
		genotype[flip_idx] = 1-genotype[flip_idx];
	}
	// Routine to check if a genotype
	// is beneficial, deleterious or neither:
	float fitness = genotype_status(genotype, parameters);
	R = 1 + fitness;
	if (R<R_min) {
		R=R_min;
	}
	if (R>R_max) {
		R=R_max;
	}
	
	outdata.gtype = genotype;
	outdata.R = R;
	outdata.mutsize = number_of_flips;
	return outdata;
}

int vec_set_diff(vector<int> vector1, vector<int> vector2) {
	sort(vector1.begin(), vector1.end());
    sort(vector2.begin(), vector2.end());
    // Initialise a vector
    // to store the differing values
    // and an iterator
    // to traverse this vector
    vector<int> v(vector1.size() + vector2.size());
    vector<int>::iterator it, st;
    it = set_difference(vector1.begin(),
                          vector1.end(),
                          vector2.begin(),
                          vector2.end(),
                          v.begin());
	return v.size();
}

float compute_multiplicity(map<int, vector<int>>  genotypes_loc) {
	int N = genotypes_loc.size();
	map<int, int> multiplicities;
	// First we add all genotypes to the same vector:
	vector<int> genotypes_all;
	for (int i = 0; i < N; i++) {
		for (int j=0; j < genotypes_loc[i].size(); j++) {
			genotypes_all.push_back(genotypes_loc[i][j]);
			multiplicities[genotypes_loc[i][j]]++;
		}
	}
	std::vector<int> value;
	for(std::map<int,int>::iterator it = multiplicities.begin(); it != multiplicities.end(); ++it) {
		value.push_back(it->second);
	}
	return ((float)accumulate(value.begin(), value.end(), 0))/((float) value.size()) ;
}

float compute_hamming_distance_within_generation(map<int, vector<int>>  genotypes_loc, int comparisons) {
	//cout << "Entering compute_hamming..." << endl;
	// Only do a predetermined number of comparisons!
	// Otherwise it scales too poorly.
	float dham = 0;
	int N = genotypes_loc.size();
	int comps_done = 0;
	while (comps_done < comparisons) {
		int i = rand() % N;
		int j = rand() % N;
		while (i == j) {
			j = rand() % N;
		}
		int dham_loc = idx_by_idx_hamming(genotypes_loc[i],genotypes_loc[j]);
		dham = dham + (float)dham_loc/((float)comparisons);
		comps_done++;
	}
	//cout << "Exiting compute_hamming..." << endl;
	return dham;
}

int count_intmap_occurrences(map<int, vector<int>> intmap,  vector<int> testvec) {
	int occurrences = 0;
	bool ident;
	for (int i = 0; i < intmap.size(); i++) {
		ident = true;
		for (int j = 0; j < intmap[i].size(); j++) {
			if (intmap[i][j] != testvec[j]) {
				ident = false;
				break;
			}
		}
		if (ident) {
			occurrences++;
		}
	}
	
	return occurrences;
}


auto compute_diversity(Params parameters, map<int, vector<int>>  genotypes_loc, int N_L, map<int, float> R0s) { // Hamming measure
	int N = genotypes_loc.size();
	float R0_initial = parameters.R0_initial;
	float eps = 0.001;
	float hammingdist = compute_hamming_distance_within_generation(genotypes_loc, 10000);
	int trackedvar_incidence = 0;
	int del_incidence = 0;
	for (int i = 0; i < R0s.size(); i++) {
		if (R0s[i] > R0_initial+eps) {
			trackedvar_incidence++;
		}
		else if (R0s[i] < R0_initial-eps) {
			del_incidence++;
		}
	}
	struct retVals {        // Declare a local structure
		float diversity;
		int trackedvar_incidence;
	};
	return retVals{hammingdist, trackedvar_incidence};

}

auto compute_hamming_distribution(map<int, vector<int>>  genotypes_loc, int N_L, map<int,vector<int>> popmembers, int pop, int comparisons) { // Hamming measure
	int N = genotypes_loc.size();
	mat hamming_distribution(1, N_L+1, fill::zeros);
	int comps_done = 0;
	int i_idx;
	int i;
	int j;
	int j_idx;
	while (comps_done < comparisons) {
		// Choose random members from population of interest! 
		i_idx = rand() % popmembers[pop].size();
		i = popmembers[pop][i_idx];
		j = i;
		while (i == j) {
			j_idx = rand() % popmembers[pop].size();
			j = popmembers[pop][j_idx];
		}
		int dham_loc = idx_by_idx_hamming(genotypes_loc[i],genotypes_loc[j]);
		hamming_distribution(dham_loc) = hamming_distribution(dham_loc) + 1;
		comps_done++;
	}
	// Finally, normalize the distribution
	hamming_distribution = hamming_distribution / accu(hamming_distribution);
	return hamming_distribution;
}

auto compute_hamming_distribution_origin(map<int, vector<int>>  genotypes_loc, vector<int> init_genotype, int N_L, map<int,vector<int>> popmembers, int pop, int comparisons) { // Hamming measure
	int N = genotypes_loc.size();
	mat hamming_distribution(1, N_L+1, fill::zeros);
	int comps_done = 0;
	int i_idx;
	int i;
	while (comps_done < comparisons) {
		// Choose random member from population of interest! 
		i_idx = rand() % popmembers[pop].size();
		i = popmembers[pop][i_idx];
		int dham_loc = idx_by_idx_hamming(genotypes_loc[i],init_genotype);
		hamming_distribution(dham_loc) = hamming_distribution(dham_loc) + 1;
		comps_done++;
	}
	// Finally, normalize the distribution
	hamming_distribution = hamming_distribution / accu(hamming_distribution);
	return hamming_distribution;
}

vector<int> generate_initial_genotype(int N_L) {
	vector<int> init_genotype;
	float r;
	for (int i=0; i < N_L; i++) {
		r = randu();
		if (r < 0.5) {
			init_genotype.push_back(1);
		}
		else {
			init_genotype.push_back(0);
		}
	}
	return init_genotype;
}

simdata run_outbreak_mutatingR(Params parameters) {
	float dR = 2;
	
	int Npops = parameters.popsizes.size();
	int stat_max = 20; // max(abs(genotype_status)) to be recorded
	mat status_dists_t(parameters.generations, 2*stat_max+1, fill::zeros);
	int n_generations = parameters.generations;
	int I_initial = parameters.I_initial;
	float R_initial = parameters.R0_initial;
	float alpha = parameters.alpha;
	int N_L = parameters.N_L;
	float k_mut = parameters.k_mut;
	
	map<int,mat> hamming_distribution;
	map<int,mat> hamming_origindist;
	map<int,mat> hamming_distributions_t;
	map<int,mat> hamming_origindists_t;
	
	// Memory conserving version!
	// Generation 1 is always the "new generation"
	// Generation 0 is always "previous generation"
	// Parameters for the mutation rate distribution (hardcoded for now):
	for (int i=0; i < Npops; i++) {
		hamming_distributions_t[i] = mat(n_generations, N_L+1, fill::zeros);
		hamming_origindists_t[i] = mat(n_generations, N_L+1, fill::zeros);
	}
	float avg_mut = 0.3;
	float beta_mut = avg_mut/k_mut;
	gamma_distribution<> mut_gam(k_mut, beta_mut);
	
	map<int, map<int, vector<int> > > genotypes; // Map containing genotypes. Structure is genotypes[generation][individual] = vector of mutations
	map<int, map<int, float> > R0s; // Map containing the R0 values for the strain carried by each individual. Structure is R0s[generation][individual] = R0;
	vector<int> generation_sizes;
	vector<int> trackedvar_incidences;
	random_device rd;
	mt19937 gen(rd());
	vector<float> diversities;
	gtype_R mutation;
	gtype_R_status mutation_and_status;
	/* Then we must initiate the genotypes and R0s for the initial generation */
	// First we create the initial genotype (random but shared among all):
	float R_initgen;
	cout << "Generating initial genotype ..." << endl;
	vector<int> init_genotype = generate_initial_genotype(N_L);
	R_initgen = gen_mut(init_genotype, R_initial, 0.0, dR, parameters).R;
	int retry_ctr = 0;
	while (R_initgen < 1.0 or R_initgen > 1.1) {
		init_genotype = generate_initial_genotype(N_L);
		R_initgen = gen_mut(init_genotype, R_initial, 0.0, dR, parameters).R;
		retry_ctr++;
		cout << "Retrying for the " << retry_ctr << "th time (R was " << R_initgen << ")" << endl;
	}
	for (int n=0; n < I_initial; n++) {
		for (int i = 0; i < N_L; i++) {
			genotypes[0][n].push_back(init_genotype[i]);
		}
		mutation = gen_mut(genotypes[0][n], R_initial, 0.0, dR, parameters);
		R0s[0][n] = mutation.R;
		if (n==0) {
			cout << "R of initial genotype was " << mutation.R << endl;
		}
		genotypes[0][n] = mutation.gtype;
	}
	cout << "Done." << endl;
	generation_sizes.push_back(R0s[0].size());
	// Insert timestep=0 popsizes vector into popsizes_t
	parameters.popsizes_t.row(0) = conv_to<rowvec>::from(parameters.popsizes);
	auto [diversity, trackedvar_incidence] = compute_diversity(parameters, genotypes[0], N_L, R0s[0]);
	cout << "Mutation diversity in generation " << 0 << " : ";
	cout << diversity  << endl;
	diversities.push_back(diversity);
	trackedvar_incidences.push_back(trackedvar_incidence);
	
	for (int i=0; i < Npops; i++) {
		hamming_distribution[i] = compute_hamming_distribution(genotypes[0], N_L, parameters.popmembers, i, 1000);
		hamming_origindist[i] = compute_hamming_distribution_origin(genotypes[0], init_genotype, N_L, parameters.popmembers, i, 1000);
		hamming_distributions_t[i].row(0) = hamming_distribution[i];
		hamming_origindists_t[i].row(0) = hamming_origindist[i];
	}
	
	int N_loc;
	int N_previous;
	int I;
	int n; 
	for (int generation=1; generation < n_generations; generation++) {
		// Separator line:
		cout << "----------------------------------------------" << endl;
		I = 0;
		N_loc = 0;
		N_previous = genotypes[0].size();
		
		vector<int> agentpop_prev = parameters.agentpop;
		map<int,vector<int>> popmembers_prev = parameters.popmembers;
		rowvec pops(parameters.popsizes.size(), fill::zeros);
		for (int j=0; j < parameters.popsizes.size(); j++) {
			pops(j)=j;
		}
		// Clear parameters.popmembers to prepare for the next generation:
		for (int j=0; j < pops.size(); j++) {
			parameters.popmembers[j].clear();
		}
		// Also clear agentpop:
		parameters.agentpop.clear();
		
		while (N_loc < N_previous) {
			n = (rand() % N_previous);
			float r = draw_single_infectivity(R0s[0][n], alpha);
			poisson_distribution<> g(r);
			int R = g(gen);
			time_t tstart, tend;
			
			int own_pop;
			int offspring_pop;
			rowvec own_inters;
			
			if (N_loc+R > N_previous) {
				R = N_previous-N_loc;
			} 
			
			int m=0;
			for (m=0; m < R; m++) {
				genotypes[1][N_loc] = genotypes[0][n];
				
				// Metapopulation
				// How we'll do it: Each person i still draws a number of
				// offspring, and the metapop of those offspring will then
				// be chosen from the different populations in proportion
				// to what's in person i's interaction matrix
				own_pop = agentpop_prev[n];
				own_inters = parameters.interpop.row(own_pop);
				// Then we need to choose a population for the newly infected:
				offspring_pop = choose_weighted_rowvecs(pops, own_inters); // variable values and weights
				parameters.agentpop.push_back(offspring_pop);
				parameters.popmembers[offspring_pop].push_back(N_loc); 
				
				float mut_rate = 0.3;
				if (k_mut < 1) { // Just a way to discern high and low heterogenity in this simplified model
					float rnum = randu();
					if (rnum < 0.0001) { // Saltation frequency
						// Jump size:
						mut_rate = parameters.mut_rate_high + 100*randu();
					}
				}
				mutation = gen_mut(genotypes[1][N_loc], R_initial, mut_rate, dR, parameters);
				R0s[1][N_loc] = mutation.R;
				genotypes[1][N_loc] = mutation.gtype;
				N_loc++;
			}
		}
		
		// Rebuild parameters.popmembers from agentpop
		for (int i = 0; i < Npops; i++) {
			parameters.popmembers[i].clear();
		}
		for (int i = 0; i < parameters.agentpop.size(); i++) {
			parameters.popmembers[parameters.agentpop[i]].push_back(i);
		}
		// Recompute the popsizes (relative population sizes)
		int N_new = R0s[1].size();
		for (int i = 0; i < parameters.popsizes.size(); i++) {
			parameters.popsizes[i] = 0.0;
		}
		for (int i = 0; i < N_new; i++) {
			parameters.popsizes[parameters.agentpop[i]] = parameters.popsizes[parameters.agentpop[i]] + 1.0/((float)N_new);
		}
		if (Npops > 1) {
			cout << "New population distribution:" << endl;
			for (int i = 0; i < Npops; i++) {
				cout << parameters.popsizes[i] << " ";
			}
			cout << endl;
		}
		// Insert current popsizes vector into popsizes_t
		parameters.popsizes_t.row(generation) = conv_to<rowvec>::from(parameters.popsizes);
		// Slow way of outputting highest R:
		float R_guess=0;
		for (int i = 0; i < R0s[1].size(); i++) {
			if (R0s[1][i] > R_guess) {
				R_guess = R0s[1][i];
			}
		}
		cout << "Highest R: " << R_guess << endl;
		
		auto [diversity_new, trackedvar_incidence_new] = compute_diversity(parameters, genotypes[1], N_L, R0s[1]);
		trackedvar_incidence = trackedvar_incidence_new;
		trackedvar_incidences.push_back(trackedvar_incidence);
		diversity = diversity_new;
		cout << "Mutation diversity in generation " << generation << " : ";
		cout << diversity << endl;
		for (int i=0; i < Npops; i++) {
			hamming_distribution[i] = compute_hamming_distribution(genotypes[1], N_L, parameters.popmembers, i, 1000);
			hamming_origindist[i] = compute_hamming_distribution_origin(genotypes[1], init_genotype, N_L, parameters.popmembers, i, 1000);
			hamming_distributions_t[i].row(generation) = hamming_distribution[i];
			hamming_origindists_t[i].row(generation) = hamming_origindist[i];
		}
		
		cout << "Average R: ";
		// First we create an R0vector, easier to handle than the map
		vector<float> R0STLvector;
		for (int n=0; n < R0s[1].size(); n++) {
		    R0STLvector.push_back(R0s[1][n]);
	    }
	    rowvec R0vec = conv_to<rowvec>::from(R0STLvector);
		cout << mean(R0vec) << " ± " << stddev(R0vec);
		cout << endl;
		diversities.push_back(diversity);
		generation_sizes.push_back(R0s[1].size());
		if (R0s[1].size() != parameters.I_initial) {
			cout << "Generation size: ";
			cout << R0s[1].size() << endl;
		}
		// Copy new generation into genotypes[0] and R0s[0] to prepare for next generation:
		R0s[0] = R0s[1];
		genotypes[0] = genotypes[1]; // Deepcopy?
	}
	simdata returndata;
	cout << "Assigning returndata." << endl;
	returndata.genotypes = genotypes;
	returndata.R0s = R0s;
	returndata.diversities = diversities; 
	returndata.generation_sizes = generation_sizes;
	returndata.trackedvar_incidences = trackedvar_incidences;
	returndata.hamming_distributions_t = hamming_distributions_t;
	returndata.hamming_origindists_t = hamming_origindists_t;
	returndata.status_dists_t = status_dists_t;
	returndata.parameters = parameters;
	cout << "Returning data to main()" << endl;
	return returndata;
}



void save_genotypes(map<int, map<int, vector<int> > > genotypes, string datadir) {
	// Output infectree:
	// Output genotypes for each generation:
	ofstream genotypes_file;
	genotypes_file.open(datadir + "genotypes.dat");
	for (int g = 0; g < genotypes.size(); g++) {	
		string outstr = "";
		outstr.append(to_string(g) + " ");
		for (int i = 0; i < genotypes[g].size() ; i++) {
			outstr.append(to_string(i) + " ");
			for (int j = 0; j < genotypes[g][i].size(); j++) {
				outstr.append(to_string(genotypes[g][i][j]) + " ");
			}
			outstr.append("\n");
			genotypes_file << outstr;
		}
	}
	genotypes_file.close();
}

void save_parameters(Params parameters, string datadir) {
	int generations = parameters.generations;
	float R0_initial = parameters.R0_initial;
	float k = parameters.alpha; // Dispersion parameter, also known as k.
	float k_mut = parameters.k_mut;
	int I_initial = parameters.I_initial; 
	ofstream parameters_file;
	parameters_file.open(datadir + "parameters.dat");
	string outstr = "";
	outstr.append("generations");
	outstr.append(" ");
	outstr.append(to_string(generations));
	outstr.append("\n");
	outstr.append("R0_initial");
	outstr.append(" ");
	outstr.append(to_string(R0_initial));
	outstr.append("\n");
	outstr.append("k");
	outstr.append(" ");
	outstr.append(to_string(k));
	outstr.append("\n");
	outstr.append("k_mut");
	outstr.append(" ");
	outstr.append(to_string(k_mut));
	outstr.append("\n");
	outstr.append("I_initial");
	outstr.append(" ");
	outstr.append(to_string(I_initial));
	outstr.append("\n");
	//
	outstr.append("N_L");
	outstr.append(" ");
	outstr.append(to_string(parameters.N_L));
	outstr.append("\n");
	//
	outstr.append("N_epitopes");
	outstr.append(" ");
	outstr.append(to_string(parameters.N_epitopes));
	outstr.append("\n");
	//
	outstr.append("N_H");
	outstr.append(" ");
	outstr.append(to_string(parameters.N_H));
	outstr.append("\n");
	//
	outstr.append("L_epitope");
	outstr.append(" ");
	outstr.append(to_string(parameters.L_epitope));
	outstr.append("\n");
	//
	outstr.append("d");
	outstr.append(" ");
	outstr.append(to_string(parameters.d));
	outstr.append("\n");
	//
	outstr.append("dR_L_mean");
	outstr.append(" ");
	outstr.append(to_string(parameters.dR_L_mean));
	outstr.append("\n");
	//
	outstr.append("dR_H_mean");
	outstr.append(" ");
	outstr.append(to_string(parameters.dR_H_mean));
	outstr.append("\n");
	//
	outstr.append("mut_rate_high");
	outstr.append(" ");
    outstr.append(to_string(parameters.mut_rate_high));
    outstr.append("\n");
	//
	
	parameters_file << outstr;
	parameters_file.close();
}

void save_diversities(vector<float> diversities, string datadir) {
	ofstream diversities_file;
	diversities_file.open(datadir + "diversities.dat");
	for (int i = 0; i < diversities.size() ; i++) {
		string outstr = "";
		outstr.append(to_string(diversities[i]) + " ");
		outstr.append("\n");
		diversities_file << outstr;
	}
	diversities_file.close();
}

void save_intvector(vector<int> intvector, string datadir, string arrname) {
	ofstream diversities_file;
	diversities_file.open(datadir + arrname + ".dat");
	for (int i = 0; i < intvector.size() ; i++) {
		string outstr = "";
		outstr.append(to_string(intvector[i]) + " ");
		outstr.append("\n");
		diversities_file << outstr;
	}
	diversities_file.close();
}

float get_R0_final(map<int, float> R0s_loc) {
	float R0mean;
	vector<float> R0STLvector;
	for (int n=0; n < R0s_loc.size(); n++) {
		R0STLvector.push_back(R0s_loc[n]);
	}
	rowvec R0vec = conv_to<rowvec>::from(R0STLvector);
	R0mean = mean(R0vec);
	
	return R0mean;
}

void save_R0data(map<int, map<int, float> > R0s, string datadir) {
	ofstream R0means_file;
	R0means_file.open(datadir + "R0means.dat");
	ofstream R0CoVs_file;
	R0CoVs_file.open(datadir + "R0CoVs.dat");
	for (int g = 0; g < 2 ; g++) {
	    vector<float> R0STLvector;
		for (int n=0; n < R0s[g].size(); n++) {
		    R0STLvector.push_back(R0s[g][n]);
	    }
	    rowvec R0vec = conv_to<rowvec>::from(R0STLvector);
	    // Means:
		string outstr = "";
		outstr.append(to_string(mean(R0vec)) + " ");
		outstr.append("\n");
		R0means_file << outstr;
		// CoVs:
		outstr = "";
		outstr.append(to_string(stddev(R0vec)/mean(R0vec)) + " ");
		outstr.append("\n");
		R0CoVs_file << outstr;
	}
	R0means_file.close();
	R0CoVs_file.close();
}

void save_seed(unsigned long seed, string datadir) {
    ofstream seed_file;
    seed_file.open(datadir + "seed.dat");
    string outstr = "";
    outstr.append(to_string(seed) + "\n");
    seed_file << outstr;
    seed_file.close();
}

mat linear_interpop(mat interpop) {
	int Npops = interpop.n_rows;
	float eps_inter = 0.0;
	for (int i = 0; i < Npops; i++) {
		for (int j = 0; j < Npops; j++) {
			if (i==j) {
				if ((i==0) or (i==Npops-1)) {
					interpop(i,j) = 1-eps_inter;
				}
				else 
				{
					interpop(i,j) = 1-2*eps_inter;
				}
			}
			else if (abs(i-j)==1) {
				interpop(i,j) = eps_inter;
			}
		}
	}
	cout << "Interaction matrix:" << endl;
	interpop.print();
	return interpop;
}

mat zerodim_interpop(mat interpop) { // "Zero dimensional" interaction matrix, in the sense that all compartments border on all others.
	int Npops = interpop.n_rows;
	float eps_inter = 0.001;
	for (int i = 0; i < Npops; i++) {
		for (int j = 0; j < Npops; j++) {
			if (i==j) {
				interpop(i,j) = 1-(Npops-1)*eps_inter;
			}
			else {
				interpop(i,j) = eps_inter;
			}
		}
	}
	cout << "Interaction matrix:" << endl;
	interpop.print();
	return interpop;
}

Params init_metapops(Params parameters) {
	int Npops = parameters.popsizes.size();
	parameters.interpop.zeros(Npops,Npops); // Interpretation:
	// g[i,j] is (relative) probability that an invididual from i will
	// infect one from j (with normalization sum(g[i,:])=1)
	
	//parameters.interpop = linear_interpop(parameters.interpop);
	parameters.interpop = zerodim_interpop(parameters.interpop);
	
	Col<float> popsizes_arma(parameters.popsizes);
	vec pops_arma(Npops, fill::zeros);
	for (int i=0; i<Npops; i++) {
		pops_arma(i)=i;
	}
	for (int i=0; i < parameters.I_initial; i++) {
		parameters.agentpop.push_back(choose_weighted(pops_arma,popsizes_arma));
		parameters.popmembers[parameters.agentpop[i]].push_back(i);
	}
	cout << "init_metapops() done" << endl;
	return parameters;
}

int main(int argc, char **argv) {
	unsigned long seed = mix(clock(), time(NULL), getpid());
	srand(seed);
	Params parameters;
	parameters.generations = 250;
	parameters.I_initial = 50000;
	parameters.R0_initial = 1.0;
	int Npops = 1;
	float eqf = 1.0/((float)Npops);
	for (int i = 0; i < Npops; i++) {
		parameters.popsizes.push_back(eqf);
	}
	parameters = init_metapops(parameters);
	parameters.popsizes_t = mat(parameters.generations, Npops, fill::zeros);
	parameters.N_L = 1000;
	parameters.alpha = 1000; // Unless set from command line
	parameters.k_mut = 0.1; // Unless set from command line
	parameters.mut_rate_high = 50; // Unless set from command line
	parameters.d = 3;
	parameters.L_epitope = 5;
	parameters.N_epitopes = 5;
	parameters.N_H = 1;
	parameters.dR_H_mean = 1.0;
	parameters.dR_L_mean = -100.0; // Unless set from command line
	
	cout << "Setting up fit configurations ... ";
	parameters.fitconfigs = setup_fitconfigs(parameters);
	cout << "Done." << endl;
	cout << "Setting up fitness values of fit configurations ... ";
	parameters.fitconfigs_fitness = setup_fitconfig_fitnesses(parameters);
	cout << "Done. " << endl;
	if (argc > 1) {
		parameters.alpha = stof(argv[1]);
		cout << "k (passed via command line) = " << parameters.alpha << endl;
		if (argc > 2) {
			parameters.k_mut = stof(argv[2]);
			cout << "k_mut (passed via command line) = " << parameters.k_mut << endl;
			if (argc > 3) {
				parameters.dR_L_mean = stof(argv[3]);
				cout << "dR_L_mean (passed via command line) = " << parameters.dR_L_mean << endl;
				if (argc > 4) {
					parameters.mut_rate_high = stof(argv[4]);
					cout << "Saltation size (passed via command line) = " << parameters.mut_rate_high << endl;
				}
			}
		}
	}
	simdata simulation = run_outbreak_mutatingR(parameters);
	cout << "Simulation routine concluded." << endl;
	
	map<int, map<int, vector<int> > > genotypes = simulation.genotypes;
	map<int, map<int, float> > R0s = simulation.R0s;
	vector<float> diversities = simulation.diversities;
	vector<int> generation_sizes = simulation.generation_sizes;
	vector<int> trackedvar_incidences = simulation.trackedvar_incidences;
	map <int,mat> hamming_distributions_t = simulation.hamming_distributions_t;
	map <int,mat> hamming_origindists_t = simulation.hamming_origindists_t;
	mat status_dists_t = simulation.status_dists_t;
	parameters = simulation.parameters;
	
	float R0_fin = get_R0_final(R0s[1]);
	cout << "Final R0 was: " << R0_fin << endl;
	
	// DATA OUTPUT
	if (R0_fin > 2.5) {
		string datadir = string("/storage/bjarke/Superspread/Phylodata/kmut_emergence") + "/data_" + to_string(seed) + "_k_" + to_string(float(parameters.alpha)) + + "_kmut_" + to_string(float(parameters.k_mut)) + "/";
		cout << "Data directory: " << datadir << endl;
		boost::filesystem::create_directories(datadir);
		cout << "Saving reproductive numbers" << endl;
		save_R0data(R0s, datadir);
		cout << "Saving diversities" << endl;
		save_diversities(diversities, datadir);
		cout << "Saving generation sizes" << endl;
		save_intvector(generation_sizes, datadir, "gensizes");
		cout << "Saving tracked variants" << endl;
		save_intvector(trackedvar_incidences, datadir, "trackedvariant");
		cout << "Saving parameters" << endl;
		save_parameters(parameters, datadir);
		for (int i=0; i < Npops; i++) {
			cout << "Saving Hamming distance distribution for each generation - population " << i << endl;
			hamming_distributions_t[i].save(datadir + "hamming_distributions" + to_string(i) + ".dat", csv_ascii);
			cout << "Saving Hamming distance (from origin) distribution for each generation - population " << i << endl;
			hamming_origindists_t[i].save(datadir + "hamming_origin_distributions" + to_string(i) + ".dat", csv_ascii);
		}
		cout << "Saving genotype statuses (deleterious, neutral, beneficial)" << endl;
		status_dists_t.save(datadir + "status_distributions.dat", csv_ascii);
		cout << "Saving population sizes." << endl;
		parameters.popsizes_t.save(datadir + "popsizes.dat", csv_ascii);
        cout << "Saving seed ... " << endl;
        save_seed(seed, datadir);
		cout << "Programme done." << endl;
	}
	return 0;
}
