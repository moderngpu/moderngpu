#include <moderngpu/kernel_intervalmove.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/kernel_segsort.hxx>
#include <fstream>
#include <sstream>

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Parse the statecity.csv file and build arrays of state segments and 
// geodetic positions.

struct state_city_t {
  struct city_t {
    int state;
    std::string city_name;
  };
  std::vector<city_t> cities;

  std::vector<std::string> states;      // Map from state codes to state names.
  std::vector<int> cities_per_state;
  std::vector<float2> city_pos;

  bool process(const char* filename) {

    std::ifstream ifs(filename);
    std::string csv;
    std::getline(ifs, csv);

    while(ifs) {
      std::string line;
      std::getline(ifs, line);
      std::istringstream iss(line);

      std::string parts[8];
      for(int i = 0; i < 8; ++i)
        std::getline(iss, parts[i], ',');
      if(2 != parts[0].size()) continue;

      city_t city;
      if(!states.size() || parts[0] != states.back()) {
        states.push_back(parts[0]);
        cities_per_state.push_back(0);
      }
      ++cities_per_state.back();

      city.state = (int)cities_per_state.size() - 1;
      city.city_name = parts[2];
      cities.emplace_back(city);

      city_pos.push_back(make_float2(
        (float)atof(parts[7].c_str()),
        (float)atof(parts[6].c_str())
      ));
    }
    std::string line;

    return cities.size() > 0;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Compute the great circle distance between two points.

template<typename real_t>
MGPU_HOST_DEVICE real_t deg_to_rad(real_t deg) {
  return (real_t)(M_PI / 180) * deg;
}

// https://en.wikipedia.org/wiki/Haversine_formula#The_haversine_formula
template<typename real_t>
MGPU_HOST_DEVICE real_t hav(real_t theta) {
  return sq(sin(theta / 2));
}

// https://en.wikipedia.org/wiki/Great-circle_distance#Computational_formulas
template<typename real_t>
MGPU_HOST_DEVICE real_t earth_distance(real_t lat_a, real_t lon_a, 
  real_t lat_b, real_t lon_b) {

  lat_a = deg_to_rad(lat_a);
  lat_b = deg_to_rad(lat_b);
  lon_a = deg_to_rad(lon_a);
  lon_b = deg_to_rad(lon_b);

  const real_t earth_radius = 3958.76;  // Approx earth radius in miles.
  real_t arg = hav(lat_b - lat_a) + 
    cos(lat_a) * cos(lat_b) * hav(lon_b - lon_a);
  real_t angle = 2 * asin(sqrt(arg));

  return angle * earth_radius;
}

////////////////////////////////////////////////////////////////////////////////
// Score 

template<int count>
struct best_t {
  struct term_t {
    float score;
    int index;
  } terms[count];

  best_t() = default;
  
  MGPU_HOST_DEVICE best_t(float score, int index) {
    fill(terms, term_t { FLT_MAX, -1 });
    terms[0] = term_t { score, index };
  }
};

struct combine_scores_t {
  template<int count>
  MGPU_HOST_DEVICE best_t<count> operator()(
    best_t<count> a, best_t<count> b) const {

    auto rotate = [](best_t<count> x) {
      best_t<count> y = x;
      iterate<count - 1>([&](int i) { y.terms[i] = y.terms[i + 1]; });
      return y;
    };
    best_t<count> c;

    iterate<count>([&](int i) {
      bool p = !(b.terms[0].score < a.terms[0].score);
      c.terms[i] = p ? a.terms[0] : b.terms[0];
      if(p) a = rotate(a); else b = rotate(b);
    });

    return c;
  }
};

////////////////////////////////////////////////////////////////////////////////

template<int d>
struct query_results {
  mem_t<best_t<d> > distances;
  mem_t<int> indices;
};

template<int d>
std::unique_ptr<query_results<d> >
compute_distances(const int* cities_per_state, const float2* city_pos, 
  const state_city_t& db, context_t& context) {

  int num_states = (int)db.states.size();
  int num_cities = (int)db.cities.size();

  // 1: Scan the state counts for state segments.
  mem_t<int> state_segments(num_states, context);
  scan(cities_per_state, num_states, state_segments.data(), context);

  // 2: Use interval_expand to create a city->city map. This 
  mem_t<int> city_to_city_map(num_cities, context);
  interval_expand(state_segments.data(), num_cities, state_segments.data(),
    num_states, city_to_city_map.data(), context);

  // 3: For each city, store the number of cities in that city's state. 
  mem_t<int> distance_segments(num_cities, context);
  int* distance_segments_data = distance_segments.data();
  transform_lbs([=]MGPU_DEVICE(int index, int seg, int rank, 
    tuple<int> state_count) {
    distance_segments_data[index] = get<0>(state_count) - 1;
  }, num_cities, state_segments.data(), num_states, 
    make_tuple(cities_per_state), context);

  // 4: Scan the number of interactions for each state for the segment markers
  // for segmented reduction. Recover the number of work-items.
  scan(distance_segments.data(), num_cities, distance_segments.data(), context);

  // 5: Define a lambda that returns the minimum distance between two cities
  // in a best_t<best_count> struct. This is fed to lbs_segreduce to find the
  // best_count-closest cities within each state.

  // Compute the number of work-items. Each city performs distance tests on
  // cities_per_state[state] - 1 other cities. The -1 is to avoid computing
  // self-distances.
  int num_work_items = 0;
  for(int count : db.cities_per_state)
    num_work_items += count * (count - 1);

  // 6. Define a lambda that returns the distance between two cities.
  // Cache state_segments_data and pos_i.
  auto compute_score = [=]MGPU_DEVICE(int index, int city_i, int rank, 
    tuple<int, float2> desc) {

    // The position of city_i is cached.
    float2 pos_i = get<1>(desc);

    // The offset of the first city in this state is cached in desc.
    // Add in the rank to get city_i. 
    int city_j = get<0>(desc) + rank;

    // Advance city_j if city_j >= city_i to avoid self-interaction.
    if(city_j >= city_i) ++city_j;

    // Load the position of city_j and compute the distance to city_i.
    float2 pos_j = city_pos[city_j];
    float distance = earth_distance(pos_i.y, pos_i.x, pos_j.y, pos_j.x);

    // Set this as the closest distance in the structure.
    return best_t<d>(distance, city_j);
  };

  // Allocate on best_t<> per result.
  std::unique_ptr<query_results<d> > results(new query_results<d>);
  results->distances = mem_t<best_t<d> >(num_cities, context);
  results->indices = mem_t<int>(num_cities, context);

  // 7. Call lbs_segreduce to fold all invocations of the 
  // Use fewer values per thread than the default lbs_segreduce tuning because
  // best_t<3> is a large type that eats up shared memory.
  best_t<d> init = best_t<d>();
  std::fill(init.terms, init.terms + d, typename best_t<d>::term_t { 0, -1 });

  lbs_segreduce<
    launch_params_t<128, 3>
  >(compute_score, num_work_items, distance_segments.data(), num_cities, 
    make_tuple(city_to_city_map.data(), city_pos), results->distances.data(), 
    combine_scores_t(), init, context);

  // 8: For each state, sort all cities by the distance of the (d-1`th) closest
  // city.  
  auto compare = []MGPU_DEVICE(best_t<d> left, best_t<d> right) {
    // Compare the least significant scores in each term.
    return left.terms[d - 1].score < right.terms[d - 1].score;
  };
  segmented_sort_indices<
    launch_params_t<128, 3> 
  >(results->distances.data(), results->indices.data(), num_cities, 
    state_segments.data(), num_states, compare, context);
 
  return results;
}

////////////////////////////////////////////////////////////////////////////////


int main(int argc, char** argv) {
  // Load the CSV database file.
  state_city_t db;
  db.process("demo/statecity.csv");

  printf("%d cities from %d states loaded.\n", 
    (int)db.cities.size(), (int)db.states.size());

  // Start a GPU context 
  standard_context_t context;

  mem_t<int> cities_per_state = to_mem(db.cities_per_state, context);
  mem_t<float2> city_pos = to_mem(db.city_pos, context);

  // Track the 3 closest cities.
  enum { d = 3 };
  auto results = compute_distances<d>(cities_per_state.data(), 
    city_pos.data(), db, context);

  // Load the data back to the host.
  std::vector<best_t<d> > distances = from_mem(results->distances);
  std::vector<int> indices = from_mem(results->indices);

  // Print the state, city, and its d-closest cities in the same state.
  auto print_city([&](best_t<d> distance, int state, int city, FILE* f) {
    fprintf(f, "%2s %15.15s", db.states[state].c_str(), 
      db.cities[city].city_name.c_str());   // print the state and city name.

    // print the names of its d-closest cities.
    for(int i = 0; i < d; ++i) {
      auto term = distance.terms[i];
      if(-1 != term.index)
        fprintf(f, "    %12.12s: %6.2f", 
          db.cities[term.index].city_name.c_str(), term.score);
    }
    fprintf(f, "\n");
  });

  int num_states = (int)db.states.size();
  for(int state = 0, city = 0; state < num_states; ++state) {
    int num_cities = db.cities_per_state[state];
    print_city(distances[city + num_cities - 1], state,
      indices[city + num_cities - 1], stdout);
    city += num_cities;
  }

  // Store the sorted distances to a file.
  FILE* f = fopen("cities_list.txt", "w");
  for(int state = 0, city = 0; state < num_states; ++state) {
    int num_cities = db.cities_per_state[state];
    for(int c = 0; c < num_cities; ++c) 
      print_city(distances[city + c], state, indices[city + c], f);
    city += num_cities;
  }
  fclose(f);
  return 0;
}


