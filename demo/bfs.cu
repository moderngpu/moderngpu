#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include "graph.hxx"

// We use the coPapersCiteseer graph from the University of Florida Sparse
// Matrix Collection.
// http://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/coPapersCiteseer.html
// Download the file in "Matrix Market" format.
const char* filename = "demo/coPapersCiteseer/coPapersCiteseer.mtx";

using namespace mgpu;

// Label vertices that have -1 value to cur_value + 1 if they are adjacent to a 
// vertex that is set tocur_value.
// Returns the size of the front for this pass.
template<typename vertices_it, typename edges_it>
int bfs(vertices_it vertices, int num_vertices, edges_it edges,
  int* values, int cur_value, context_t& context) {

  // Allocate space for load-balancing search segments and total work-items.
  mem_t<int> segments(num_vertices, context);
  mem_t<int> count(1, context);

  // Compute the number of neighbors to explore for each vertex.
  // Emit 0 items if the vertex is not active this pass, otherwise
  // emit vertices[vertex + 1] - vertices[vertex] items, indicating that we
  // need to visit all of the vertex's neighbors.

  // Scan the number of neighbors for each vertex on the frontier.
  auto segment_sizes = [=]MGPU_DEVICE(int vertex) {
    int count = 0;
    if(values[vertex] == cur_value) {
      int begin = vertices[vertex];
      int end = vertices[vertex + 1];
      count = end - begin;
    }
    return count;
  };
  transform_scan<int>(segment_sizes, num_vertices, segments.data(), 
    plus_t<int>(), count.data(), context);

  // Read out the number of work-items and quit if there are none. That means
  // we've visited all vertices connected to a source.
  int front = from_mem(count)[0];
  if(!front) return 0;

  // Compare-and-swap cur_value + 1 into values[edges[vertices[seg] + rank]]].
  // get<0>(v) is a cache-optimized copy of vertices[seg]. 
  auto update = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> v) {
    // Compare and swap in cur_value + 1 when the old value was -1.
    int neighbor = edges[get<0>(v) + rank];
    atomicCAS(values + neighbor, -1, cur_value + 1);
  };
  transform_lbs(update, front, segments.data(), num_vertices, 
    make_tuple(vertices), context);

  return front;
}

int main(int argc, char** argv) {
  std::unique_ptr<graph_t> graph = load_graph(filename);

  standard_context_t context;
  mem_t<int> vertices = to_mem(graph->edge_indices, context);
  mem_t<int> edges = to_mem(graph->edges, context);
  int num_vertices = graph->num_vertices;
  int num_edges = graph->num_edges;
  printf("NUM VERTICES = %d    NUM_EDGES = %d\n", num_vertices, num_edges);

  // Allocate space for the vertex levels.
  // Set the source to vertex 23.
  std::vector<int> values_host(num_vertices, -1);
  values_host[23] = 0;      // 0 indicates a source.

  mem_t<int> values = to_mem(values_host, context);

  for(int cur_level = 0; ; ++cur_level) {
    int front = bfs(vertices.data(), num_vertices, edges.data(),
      values.data(), cur_level, context);
    printf("Front for level %d has %d edges.\n", cur_level, front);
    if(!front) break;
  }

  // Print the results to a file.
  FILE* f = fopen("bfs.txt", "w");
  values_host = from_mem(values);
  for(int i = 0; i < num_vertices; ++i)
    fprintf(f, "%5d: %4d\n", i, values_host[i]);
  fclose(f);
    
  return 0;
}
