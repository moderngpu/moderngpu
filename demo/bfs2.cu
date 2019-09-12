#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_workcreate.hxx>     // dynamic work creation.
#include "graph.hxx"

// We use the coPapersCiteseer graph from the University of Florida Sparse
// Matrix Collection.
// http://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/coPapersCiteseer.html
// Download the file in "Matrix Market" format.
const char* filename = "demo/coPapersCiteseer/coPapersCiteseer.mtx";

using namespace mgpu;

struct workload_t {
  int count;
  int num_segments;
  mem_t<int> segments;        // scanned sum of active vertex edge counts.
  mem_t<int> edge_indices;    // indices into edges array.
};

// Label vertices that have -1 value to cur_value + 1 if they are adjacent to a 
// vertex that is set tocur_value.
// Returns the size of the front for this pass.
template<typename vertices_it, typename edges_it>
void bfs2(vertices_it vertices, int num_vertices, edges_it edges, 
  int* values, int cur_value, workload_t& wl, context_t& context) {

  // Create a dynamic work-creation engine.
  auto engine = expt::lbs_workcreate(wl.count, wl.segments.data(), 
    wl.num_segments, context);

  // The upsweep attempts atomicCAS. If it succeeds, return the number of 
  // edges for that vertex.
  auto wl2_count = engine.upsweep(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> desc) {
      int neighbor = edges[get<0>(desc) + rank];
      int count = 0;
      if(-1 == atomicCAS(values + neighbor, -1, cur_value + 1))
        count = vertices[neighbor + 1] - vertices[neighbor];
      return count;
    }, make_tuple(wl.edge_indices.data())
  );

  // The downsweep streams out the new edge pointers.
  mem_t<int> edge_indices(wl2_count.num_segments, context);
  mem_t<int> segments = engine.downsweep(
    [=]MGPU_DEVICE(int dest_seg, int index, int seg, int rank, 
      tuple<int> desc, int* out_edge_indices) {

      // Return the same count as before and store output segment-specific
      // data using dest_index.
      int neighbor = edges[get<0>(desc) + rank];
      int begin = vertices[neighbor];
      int end = vertices[neighbor + 1];

      out_edge_indices[dest_seg] = begin;
      return end - begin;
    }, make_tuple(wl.edge_indices.data()), edge_indices.data()
  );

  // Update the workload.
  wl.count = wl2_count.count;
  wl.num_segments = wl2_count.num_segments;
  wl.segments = std::move(segments);
  wl.edge_indices = std::move(edge_indices);
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
  int source = 23;
  values_host[source] = 0;      // 0 indicates a source.
  mem_t<int> values = to_mem(values_host, context);

  // Start up the first level of the workload.
  int source_begin = graph->edge_indices[source];
  int source_end = graph->edge_indices[source + 1];
  workload_t wl;
  wl.count = source_end - source_begin;
  wl.num_segments = 1;

  std::vector<int> edge_indices_host = { source_begin };
  wl.segments = fill<int>(0, 1, context);
  wl.edge_indices = to_mem(edge_indices_host, context);

  for(int cur_level = 0; ; ++cur_level) {
    bfs2(vertices.data(), num_vertices, edges.data(), values.data(), 
      cur_level, wl, context);
    printf("Front for level %d has %d vertices and %d edges.\n", 
      cur_level, wl.num_segments, wl.count);
    if(!wl.num_segments) break;
  }

  // Print the results to a file.
  FILE* f = fopen("bfs.txt", "w");
  values_host = from_mem(values);
  for(int i = 0; i < num_vertices; ++i)
    fprintf(f, "%5d: %4d\n", i, values_host[i]);
  fclose(f);
    
  return 0;
}

