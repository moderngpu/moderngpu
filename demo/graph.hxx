#pragma once
#include <vector>
#include <memory>

struct graph_t {
  int num_vertices;
  int num_edges;
  std::vector<int> edge_indices;
  std::vector<int> edges;
};

std::unique_ptr<graph_t> load_graph(const char* name);
