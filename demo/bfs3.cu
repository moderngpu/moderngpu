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

// Visit all edges for vertices in the frontier and described in the workload_t
// structure. Overwrite this with a new workload_t structure for the next
// level in the algorithm.
// Return the number of vertices in the next level of the frontier and stream
// their IDs to frontier_vertices.

template<typename vertices_it, typename edges_it>
int bfs3(vertices_it vertices, edges_it edges, int* visited_bits, 
  int* frontier_vertices, workload_t& wl, context_t& context) {

  // Create a dynamic work-creation engine.
  auto engine = expt::lbs_workcreate(wl.count, wl.segments.data(), 
    wl.num_segments, context);

  // The upsweep attempts atomicOr. If it succeeds, return the number of 
  // edges for that vertex.
  auto wl2_count = engine.upsweep(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> desc) {
      int count = 0;
      int neighbor = edges[get<0>(desc) + rank];
      int mask = 1<< (31 & neighbor);
      if(0 == (mask & atomicOr(visited_bits + neighbor / 32, mask)))
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
      // data using dest_seg.
      int neighbor = edges[get<0>(desc) + rank];
      int begin = vertices[neighbor];
      int end = vertices[neighbor + 1];

      // Store the pointer into the edges array for the new work segment.
      out_edge_indices[dest_seg] = begin;

      // Stream out the vertex index.
      frontier_vertices[dest_seg] = neighbor;

      return end - begin;
    }, make_tuple(wl.edge_indices.data()), edge_indices.data()
  );

  // Update the workload.
  wl.count = wl2_count.count;
  wl.num_segments = wl2_count.num_segments;
  wl.segments = std::move(segments);
  wl.edge_indices = std::move(edge_indices);

  return wl.num_segments;
}


int main(int argc, char** argv) {
  std::unique_ptr<graph_t> graph = load_graph(filename);

  standard_context_t context;
  mem_t<int> vertices = to_mem(graph->edge_indices, context);
  mem_t<int> edges = to_mem(graph->edges, context);
  int num_vertices = graph->num_vertices;
  int num_edges = graph->num_edges;

  printf("NUM VERTICES = %d    NUM_EDGES = %d\n", num_vertices, num_edges);

  // Allocate space for the vertex visited bits and vertex indices.
  mem_t<int> visited_bits = fill<int>(0, div_up(num_vertices, 32), context);
  mem_t<int> ids_by_level(num_vertices, context);
  std::vector<int> counts_by_level;

  int* visited_bits_data = visited_bits.data();
  int* ids_by_level_data = ids_by_level.data();

  // Set the source to vertex 23.
  int source = 23;
  transform([=]MGPU_DEVICE(int index) {
    // Set the source vertex.
    visited_bits_data[source / 32] = 1<< (31 & source);

    // Stream out the source vertex Id.
    ids_by_level_data[0] = source;
  }, 1, context);
  counts_by_level.push_back(1);
  ids_by_level_data += 1;

  // Start up the first level of the workload.
  int source_begin = graph->edge_indices[source];
  int source_end = graph->edge_indices[source + 1];
  workload_t wl;
  wl.count = source_end - source_begin;
  wl.num_segments = 1;

  std::vector<int> edge_indices_host = { source_begin };
  wl.segments = fill<int>(0, 1, context);
  wl.edge_indices = to_mem(edge_indices_host, context);

  // After completion of this loop, ids_by_level contains the IDs of all
  // connected vertices sorted by distance from the source.
  for(int cur_level = 0; ; ++cur_level) {
    int streamed = bfs3(vertices.data(), edges.data(), visited_bits_data, 
      ids_by_level_data, wl, context);

    printf("Front for level %d has %d vertices and %d edges.\n", 
      cur_level, streamed, wl.count);
    ids_by_level_data += streamed;
    counts_by_level.push_back(streamed);
    if(!wl.num_segments) break;
  }

  return 0;
}
