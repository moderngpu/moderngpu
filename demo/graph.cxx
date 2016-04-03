#include "graph.hxx"
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cassert>

std::unique_ptr<graph_t> load_graph(const char* name) {

  FILE* f = fopen(name, "r");
  if(!f) return false;

  char line[100];
  while(fgets(line, 100, f) && '%' == line[0]);

  int height, width, num_edges;
  if(3 != sscanf(line, "%d %d %d", &height, &width, &num_edges)) {
    printf("Error reading %s\n", name);
    exit(0);
  }
  int num_vertices = height;

  std::vector<std::pair<int, int> > pairs(2 * num_edges);
  for(int edge = 0; edge < num_edges; ++edge) {
    std::pair<int, int> pair;
    if(!fgets(line, 100, f) || 
      2 != sscanf(line, "%d %d", &pair.first, &pair.second)) {
      printf("Error reading %s\n", name);
      exit(0);
    }

    // The Matrix Market format uses 1-based indexing. Decrement the vertex
    // ID so its sensible to the rest of our code.
    --pair.first, --pair.second;
    pairs[edge] = pair;
    std::swap(pair.first, pair.second);
    pairs[edge + num_edges] = pair;
  }
  num_edges *= 2;

  // Sort the pairs.
  std::sort(pairs.begin(), pairs.end());

  // Insert the edges.
  std::vector<int> edge_indices(num_vertices + 1, num_edges);
  std::vector<int> edges(num_edges);
  int cur_vertex = -1;
  for(int edge = 0; edge < num_edges; ++edge) {
    while(cur_vertex < pairs[edge].first) 
      edge_indices[++cur_vertex] = edge;
    edges[edge] = pairs[edge].second;
  }

  return std::unique_ptr<graph_t>(
    new graph_t { num_vertices, num_edges, edge_indices, edges }
  );
}
