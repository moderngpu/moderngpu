#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

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
int bfs(vertices_it vertices, int num_vertices, edges_it edges, int num_edges,
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
      int end = (vertex + 1 < num_vertices) ? vertices[vertex + 1] : num_edges;
      count = end - begin;
    }
    return count;
  };
  transform_scan(segment_sizes, num_vertices, segments.data(), plus_t<int>(), 
    count.data(), context);

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

bool load_graph(const char* name, std::vector<int>& vertices, 
  std::vector<int>& edges) {

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
    pairs[edge] = pair;
    std::swap(pair.first, pair.second);
    pairs[edge + num_edges] = pair;
  }
  num_edges *= 2;

  // Sort the pairs.
  std::sort(pairs.begin(), pairs.end());

  // Insert the edges.
  vertices.resize(num_vertices, num_edges);
  edges.resize(num_edges);
  int cur_vertex = -1;
  for(int edge = 0; edge < num_edges; ++edge) {
    while(cur_vertex < pairs[edge].first) vertices[++cur_vertex] = edge;
    edges[edge] = pairs[edge].second;
  }

  return true;
}

int main(int argc, char** argv) {
  std::vector<int> vertices_host, edges_host;
  bool success = load_graph(filename, vertices_host, edges_host);

  standard_context_t context;
  mem_t<int> vertices = to_mem(vertices_host, context);
  mem_t<int> edges = to_mem(edges_host, context);
  int num_vertices = (int)vertices.size();
  int num_edges = (int)edges.size();

  // Allocate space for the vertex levels.
  // Set the source to vertex 23.
  std::vector<int> values_host(num_vertices, -1);
  values_host[23] = 0;      // 0 indicates a source.

  mem_t<int> values = to_mem(values_host, context);

  for(int cur_level = 0; ; ++cur_level) {
    int front = bfs(vertices.data(), num_vertices, edges.data(), num_edges,
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
