# Examples of automatic load-balancing

In the past CUDA programmers were responsible for decomposing their own problem over the regular grid of GPU threads. Let's say you need to reduce values from many different non-uniformly sized segments. Which processing units load and reduce which values? Do you assign one thread to a segment, or a warp's worth or a block's worth of threads? If you settle on a vector width and the variance of the segment size is large, do you waste vector processors by assigning small segments to a large width; do you create load-imbalance by oversubscribing large segments to too-small vectors?

moderngpu 1.0 introduced the _load-balancing search_ algorithm for dynamically mapping work to processors in a way that doesn't starve any processors of work or oversubscribe work to too-few cores. However there was a considerable cost to using this mechanism: you had to write your own kernel, allocate shared memory for the load-balancing search, and communicate between threads. In the current version, the composition model has changed, and the high-level transformers take care of all of this. You just provide a lambda and the library calls it with the desired segment information.

## Sparse matrix * vector 
Features demonstrated:

1. `transform_segreduce`

**`kernel_segreduce.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename matrix_it,
  typename columns_it, typename vector_it, typename segments_it, 
  typename output_it>
void spmv(matrix_it matrix, columns_it columns, vector_it vector,
  int count, segments_it segments, int num_segments, output_it output,
  context_t& context) { 

  typedef typename std::iterator_traits<matrix_it>::value_type type_t;

  transform_segreduce<launch_arg_t>([=]MGPU_DEVICE(int index) {
    return matrix[index] * ldg(vector + columns[index]);    // sparse m * v.
  }, count, segments, num_segments, output, plus_t<type_t>(), 
   (type_t)0, context);
}
```
In moderngpu 1.0 the sparse matrix * dense vector operator, `spmv`, needed a lot of custom programming. But in moderngpu 2.0 it's just an expression of the `transform_segreduce` pattern. The user provides a CSR-style segments-descriptor array and an MGPU_DEVICE-tagged lambda to the transformer. This lambda is called once per input element, where it loads one non-zero matrix element, looks up the column index of that matrix element, gathers a vector element at that location, and returns the product. 

`transform_segreduce` is a very intelligent transformer. It automatically distributes the work of function evaluation over the regular grid of GPU threads in a geometry-oblivious way. Matrices with a very large number of non-zero elements per row parallelize just as well as much sparser matrices. Matrices with a large variance in the number of non-zero elements per row parallelize just as well as more uniform systems.

This new `spmv` function is efficient, robust, flexible and tunable. Matrices with empty rows are handled just fine--the empty rows of the output are plugged with 0 (although you could pass in any value). The `__ldg` CUDA intrinsic is used on `sm_35` and later architectures, but only when `vector` is a pointer to an arithmetic type, so you can still pass in an iterator and generate the vector values on-the-fly. 

The tuning of the function is also pushed out to the user: pass a `launch_arg_t` to specialize the kernel over an architecture-specific set of launch parameters, including the block size, grain size, cache ratio and SM occupancy. By moving tuning parameters _outside_ of the function implementation, the user can tweak the parameters for best performance for their specific needs, including generating multiple versions of the same function for inputs with different characteristics.

## Interval move and interval expand

Features demonstrated:

1. `transform_lbs`

Interval expand and interval move were introduced in moderngpu 1.0 and written as kernels that used the CTA-level load-balancing search components. `interval_expand` is like a batched `std::fill`; it sets all the elements in each segment with a single segment-specific value. `interval_move` is like a batched `std::copy`; for each segment it copies a sequence of data from a gather point to a scatter point.

```cpp
template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename output_it>
void interval_expand(input_it input, int count, segments_it segments,
  int num_segments, output_it output, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;
  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank) {
      output[index] = input[seg];
    }, 
    count, segments, num_segments, context
  );
}

template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank) {
      output[scatter[seg] + rank] = input[gather[seg] + rank];
    }, 
    count, segments, num_segments, context
  );
}
```
Both functions are succinctly implemented with the new `transform_lbs` high-level transformer. The user-provided lambda is called once for each work-item and passed the needed context: the index of the work-item within the global operation, the segment of the work-item, and the rank of the work-item within the segment. In the case of `interval_expand`, `index = 110`, `seg = 4`, `rank = 20` means we need to fill `output[110]` with the value `input[seg]`. For `interval_move` it means we need to copy the 20th item of segment 4 from `input[gather[seg] + rank]` to `output[scatter[seg] + rank]`.

`transform_lbs` with user-provided lambdas is the natural language for segmented operations like `interval_move`. Indeed, the code is a more readable than a textual description. But there is one drawback to the implementations above; they aren't as fast as the hand-coded versions in moderngpu 1.0. You see that the fill-value `input` array is dereferenced once for each item in the same segment in `interval_expand`, and both `scatter` and `gather` are dereferenced for each item in the same segment in `interval_move`. These are redundant loads and they eat up L2 bandwidth. The hand-coded version cooperatively loaded these terms by segment ID into shared memory, then read the cached values out when storing to output or computing indexing.

Sacrificing efficiency for convenience is a hard sell, especially in HPC software. How can we cache values that are indexed per-segment and retain the convenience of the `transform_lbs` interface?

**`kernel_intervalmove.hxx`**
```cpp
template<typename launch_arg_t = empty_t, typename input_it, 
  typename segments_it, typename output_it>
void interval_expand(input_it input, int count, segments_it segments,
  int num_segments, output_it output, context_t& context) {

  typedef typename std::iterator_traits<input_it>::value_type type_t;
  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<type_t> desc) {
      output[index] = get<0>(desc);
    }, 
    count, segments, num_segments, make_tuple(input), context
  );
}

template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int, int> desc) {
      output[get<0>(desc) + rank] = input[get<1>(desc) + rank];
    }, 
    count, segments, num_segments, make_tuple(scatter, gather), context
  );
}
```
`transform_lbs` and the related `lbs_segreduce` functions are overloaded to take optional `mgpu::tuple<>` parameters. `make_tuple` fuses together _cacheable iterators_ pointing to per-segment data. The caching overload of `transform_lbs` loads from the cacheable iterators through shared memory and passes the values to the lambda, as its tuple-valued fourth parameter.

In the improved `interval_move` implementation, the scatter and gather offsets, which are likely `const int*` types, are fused into a `tuple<const int*, const int*>` value and passed to `transform_lbs`. This high-level transformer cooperatively loads one value per tied iterator for each segment spanned by the CTA through shared memory and out into register. These cached values are constructed into a `tuple<int, int>` and passed to the lambda. `interval_move`'s lambda uses `mgpu::get<>` to extract the scatter and gather offsets, which it adds to the work-item's rank as in the uncached implementation.

Explicit caching of per-segment values is cooperatively parallel operation, but `interval_expand` and `interval_move` include no cooperatively parallel code. They benefit from the feature by way of this improved composition model.

## Relational join
Features demonstrated:

1. `sorted_search`
1. `transform_scan`
1. `transform_lbs`

Consider table A joined with table B. We want to record the outer product of all matching keys from A and B.

| index| table A | table B     |
|------|---------|-------------|
| 0    | ape     | chicken     |
| 1    | ape     | cow         |
| 2    | kitten  | goat        |
| 3    | kitten  | kitten      |
| 4    | kitten  | kitten      |
| 5    | zebra   | tiger       |
| 6    |         | zebra       |

inner join table:

| index | A index  | A key      | B index | B key      |
|-------|----------|------------|---------|------------|
| 0     | 2        | kitten (0) | 3       | kitten (0) |
| 1     | 2        | kitten (0) | 4       | kitten (1) |
| 2     | 3        | kitten (1) | 3       | kitten (0) |
| 3     | 3        | kitten (1) | 4       | kitten (1) |
| 4     | 4        | kitten (2) | 3       | kitten (0) |
| 5     | 4        | kitten (2) | 4       | kitten (1) |
| 6     | 5        | zebra (0)  | 6       | zebra (0)  |

Joins are a backbone of data science and analytics. They're something we want in the new version and they're something we had in 1.0. But the high-level transforms allow for clean and efficient implementations without writing new kernels.

| index | key    | lower bound| upper bound |upper - lower| scan(upper - lower)|
|-------|--------|-------------|------------|-------------|--------------------|
|     0 | ape    | 0           | 0          | 0           | 0                  |
|     1 | ape    | 0           | 0          | 0           | 0                  |
|     2 | kitten | 3           | 5          | 2           | 0                  |
|     3 | kitten | 3           | 5          | 2           | 2                  |
|     4 | kitten | 3           | 5          | 2           | 4                  |
|     5 | zebra  | 6           | 7          | 1           | 6                  |
|       |        |             |            |             | 7                  |

First we call `sorted_search` twice: once to find the lower-bound and once to find the upper-bound of each element in A in B. This is a "find sorted needles in a sorted haystack" operation. The element-wise difference of the upper- and lower-bound indices gives the number of keys in B that match each key in A. We scan this difference to produce a segments-descriptor for load-balancing search, and use `transform_lbs` to construct an `int2` array with the (A index, B index) pairs of the inner join.

**`kernel_join.hxx`**
```cpp
template<typename launch_arg_t = empty_t, 
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count, 
  comp_t comp, context_t& context) {

  // Compute lower and upper bounds of a into b.
  mem_t<int> lower(a_count, context);
  mem_t<int> upper(b_count, context);
  sorted_search<bounds_lower, launch_arg_t>(a, a_count, b, b_count, 
    lower.data(), comp, context);
  sorted_search<bounds_upper, launch_arg_t>(a, a_count, b, b_count, 
    upper.data(), comp, context);

  // Compute output ranges by scanning upper - lower. Retrieve the reduction
  // of the scan, which specifies the size of the output array to allocate.
  mem_t<int> scanned_sizes(a_count, context);
  const int* lower_data = lower.data();
  const int* upper_data = upper.data();

  mem_t<int> count(1, context);
  transform_scan([=]MGPU_DEVICE(int index) {
    return upper_data[index] - lower_data[index];
  }, a_count, scanned_sizes.data(), plus_t<int>(), count.data(), context);

  // Allocate an int2 output array and use load-balancing search to compute
  // the join.
  int join_count = from_mem(count)[0];
  mem_t<int2> output(join_count, context);
  int2* output_data = output.data();

  // Use load-balancing search on the segmens. The output is a pair with
  // a_index = seg and b_index = lower_data[seg] + rank.
  auto k = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int> lower) {
    output_data[index] = make_int2(seg, get<0>(lower) + rank);
  };
  transform_lbs<launch_arg_t>(k, join_count, scanned_sizes.data(), a_count,
    make_tuple(lower_data), context);

  return output;
}
```
The new `inner_join` implementation joins two tables as represented by sorted arrays. The arrays need not consist of arithmetic types; they only need to be comparable using the passed-in `comp` object, which returns `a < b`.

`sorted_search` takes a `bounds_lower` or `bounds_upper` argument for specializing on the search type. We then scan `a_count` sizes, which are element-wise differences of `upper` and `lower` as passed in through a lambda to the moderngpu `transform_scan` function. The reduction of this scan is returned into GPU-memory, and we load this count into a variable on the host where it's used for allocating just the right amount of memory for the join.

Once again `transform_lbs` provides the parallel intelligence for this operation. The indices for each pair of the inner join are:
```
a_index = seg;
b_index = lower_data[seg] + rank;
```
As in the `interval_move` example we choose to cache `lower_data[seg]` with the tuple-caching mechanism. 

This join implementation is easy to write, obvious on inspection, and robust with respect to the shape of the join. Running-time is linear in the sum of the two load-balancing search arguments: the join count and the size of table A.

## Breadth-first search

Features demonstrated:

1. `transform_scan`
2. `transform_lbs`

**Note:** To run this demo you'll need to download the [coPapersCiteseer](http://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/coPapersCiteseer.html) graph in the _Matrix Market_ format from the University of Florida Sparse Matrix Collection. Decompress the `.mtx` file into `demo/coPapersCiteseer/coPapersCiteseer.mtx`.

#### `bfs.cu`

```cpp
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

```
```
NUM VERTICES = 434102    NUM_EDGES = 32073440
Front for level 0 has 163 edges.
Front for level 1 has 21112 edges.
Front for level 2 has 9561 edges.
Front for level 3 has 98990 edges.
Front for level 4 has 900032 edges.
Front for level 5 has 4617596 edges.
Front for level 6 has 15439710 edges.
Front for level 7 has 8686685 edges.
Front for level 8 has 1781177 edges.
Front for level 9 has 365748 edges.
Front for level 10 has 113306 edges.
Front for level 11 has 24539 edges.
Front for level 12 has 7215 edges.
Front for level 13 has 5198 edges.
Front for level 14 has 1871 edges.
Front for level 15 has 427 edges.
Front for level 16 has 18 edges.
Front for level 17 has 17 edges.
Front for level 18 has 15 edges.
Front for level 19 has 12 edges.
Front for level 20 has 5 edges.
Front for level 21 has 10 edges.
Front for level 22 has 6 edges.
Front for level 23 has 11 edges.
Front for level 24 has 15 edges.
Front for level 25 has 1 edges.
Front for level 26 has 0 edges.
```
The moderngpu 2.0 demo "bfs" includes another practical algorithm has its load-balancing concerns satisfied by the high-level transforms. [Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) computes the minimum distance measured in edges for each vertex in a graph to one or more sources. This isn't an algorithmically efficient implementation--that would require clever management of the vertex frontier. But this is a load-balanced parallel implementation and it helps to demonstrate the facility of `transform_lbs`.

Breadth-first search updates an array of unweighted distances. Source vertices have distance 0. Visited vertices are marked by the number of edges to the closest source, and unvisited vertices are marked -1. This implementation of `bfs` updates the values array by marking unvisited vertices that are one edge away from vertices marked as `cur_value`.

Graphs are often encoded with compressed sparse row segment descriptors. The format used to encode sparse matrices helpfully share a format with the segment-descriptors arrays in moderngpu. Although we could feed this segments-descriptor array directly to the load-balancing search, we want to filter it to only process vertices on the frontier. We run `transform_scan` on the number of _active edges_ for each vertex at each step: 0 edges if the vertex _is not_ on the frontier at that step (i.e. `values[vertex] != cur_value`) or its full number of edges if the vertex _is_ on the frontier.

`transform_lbs` calls the provided lambda once for each edge attached to an active vertex. The high-level transform is robust with respect to the degree of the vertex and to the variance of the degree. We use the tuple segment caching mechanism to load the vertex's offset into the edge array (`tuple<int> v`), add in the rank of the edge within the vertex, and load from the `edge` array. This gives us the index of the neighboring vertex. The CUDA intrinsic `atomicCAS` sets the unweighted distance if and only if that vertex is unvisited.

A starting point for a more efficient BFS algorithm would be to use the return value of `atomicCAS` to make a list of vertices on the active front (to reduce or eliminate the cost of the `transform_scan` which operates over all vertices in the entire graph) or to remove some or all occurrences of visited vertices from the array of edges. moderngpu includes useful functions like `bulk_insert` and `bulk_remove` for the parallel array-surgery operations needed for more advanced bfs algorithms. 

## Attu Station, AK

Features demonstrated:

1. `scan`
2. `interval_expand`
3. `lbs_segreduce`
4. `segmented_sort` 

#### `cities.cu`

 ```
29510 cities from 52 states loaded.
AL          Phenix         Ladonia:   4.21    Smiths Stati:   7.12         Opelika:  24.78
AK    Attu Station            Adak: 435.04            Atka: 527.73        St. Paul: 712.48
AZ          Scenic     Littlefield:   6.58      Beaver Dam:   8.91    Grand Canyon:  55.75
AR        Junction       El Dorado:  13.95        Norphlet:  20.89          Strong:  21.88
CA         Needles       Bluewater:  48.57       Big River:  48.91          Blythe:  80.15
CO        Dinosaur         Rangely:  16.66         Maybell:  52.09          Meeker:  60.25
CT          Sharon       Lakeville:   5.79           Falls:   7.67          Canaan:  12.95
DE          Delmar          Laurel:   6.95          Bethel:   7.88          Blades:  11.67
DC      Washington
FL    Steinhatchee           Cross:  15.98    Horseshoe Be:  17.70    Fanning Spri:  27.97
GA           Fargo      Homerville:  26.30     Statenville:  26.62          Argyle:  27.09
HI        Maunaloa        Kualapuu:   9.65      Kaunakakai:  13.89        Ualapu'e:  24.68
ID          Salmon         Leadore:  43.09         Challis:  49.02         Clayton:  67.93
IL      Metropolis       Brookport:   3.75           Joppa:   9.71         Belknap:  18.32
IN    Mount Vernon    Parkers Sett:  12.30     New Harmony:  13.37      Poseyville:  17.20
IA       New Albin         Lansing:  10.03          Waukon:  18.45      Waterville:  19.95
KS        Kanorado        Goodland:  17.30     St. Francis:  32.57          Weskan:  32.77
KY          Albany     Burkesville:  14.43      Monticello:  18.84       Jamestown:  21.32
LA      Grand Isle    Golden Meado:  19.14        Galliano:  23.26    Port Sulphur:  27.35
ME         Houlton          Blaine:  26.14       Mars Hill:  27.54    Presque Isle:  39.94
MD      Manchester       Hampstead:   3.35     Westminster:   8.23     New Windsor:  13.64
MA      Siasconset       Nantucket:   6.42         Madaket:  11.12         Chatham:  28.24
MI        Newberry       St. James:  42.24      Manistique:  44.75      St. Ignace:  49.28
MN    Grand Marais          Lutsen:  17.21         Finland:  48.10      Silver Bay:  54.10
MS Alcorn State Un     Port Gibson:  10.63         Fayette:  12.42      Morgantown:  24.41
MO           Salem    Edgar Spring:  18.77         Licking:  20.31          Bunker:  21.72
MT          Jordan       Fort Peck:  51.91          Nashua:  61.74         Glasgow:  62.00
NE         Hyannis          Arthur:  29.83          Mullen:  36.97          Seneca:  47.89
NV          Owyhee    Fort McDermi:  73.38           Osino:  73.93    Paradise Val:  74.52
NH West Stewartsto       Colebrook:   6.84        Groveton:  26.20       Lancaster:  34.75
NJ     Port Norris     Laurel Lake:   5.28      Belleplain:   8.98       Millville:   9.67
NM         Clayton       Grenville:  27.74      Des Moines:  43.95          Folsom:  50.95
NY      Speculator     North Creek:  20.25       Long Lake:  26.53      Northville:  26.74
NC       Engelhard       Fairfield:  12.46    Swan Quarter:  18.68        Hatteras:  27.15
ND         Grenora         Fortuna:  21.27           Alamo:  21.55         Ambrose:  31.14
OH           Crown         Athalia:   5.60    Proctorville:  11.46      Chesapeake:  14.15
OK          Kenton            Felt:  25.51           Boise:  27.89           Keyes:  40.02
OR   Jordan Valley          Adrian:  52.62           Nyssa:  62.16          Harper:  67.03
PA       Driftwood        Emporium:  12.85    Prospect Par:  12.89       Pine Glen:  17.93
RI          Greene       Clayville:   4.82    Foster Cente:   5.93         Wyoming:  13.17
SC  McClellanville         Awendaw:  13.11       Jamestown:  19.00      Georgetown:  22.53
SD      Camp Crook         Buffalo:  21.02         Prairie:  56.03    Belle Fourch:  61.49
TN      Copperhill        Ducktown:   2.58          Benton:  19.10          Etowah:  24.69
TX        Van Horn    Sierra Blanc:  31.49       Valentine:  37.15    Fort Hancock:  62.19
UT        Wendover          Dugway:  74.97     Grantsville:  80.77     Rush Valley:  87.19
VT   Beecher Falls          Canaan:   1.93     Island Pond:  23.69      Derby Line:  30.30
VA        Monterey       Deerfield:  17.54     Craigsville:  24.99    Augusta Spri:  25.11
WA        Oroville          Loomis:  12.83        Tonasket:  16.40       Riverside:  30.16
WV      Brandywine        Franklin:   5.31         Whitmer:  21.26          Harman:  25.85
WI        Tomahawk     Rhinelander:  17.60         Merrill:  20.62    Lake Tomahaw:  23.72
WY         Mammoth            Alta:  84.55            Cody:  86.25         Ralston:  90.44
PR         Culebra         Vieques:  14.52       Esperanza:  18.53     Las Croabas:  21.85
 ```

The `cities.cu` demo implements a geographic data sciences query, not too far off from what a skilled SQL user might formulate. We start with a CSV file with American cities and their geodetic locations, sorted by US state. The goal is to find the three closest cities in each state for every city, as a sort of 'remoteness' metric. (3 is a template argument here. Any number will work but comparison costs increase quadratically.) The cities are then sorted by the distance to their third-closest city. This is a more robust measure of remoteness than taking simply the shortest pair. Two "census-designated places" may be close together yet very far from any other place. This measure is robust for up to three places in a cluster out by themselves--the third closest to each of those will be outside of the cluster and give truer sense of "remoteness." 

There are a limitless number of reasonable definitions of a remoteness measure: take the average distance to all other cities in the state, take the sum of the squared distances to all other cities, etc. The beauty of using moderngpu's high-level transformers is that most of the client code remains the same between these different metrics and only the reduction operator changes.

```cpp
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
```
Here we define the distance storage type `best_t<>`. If `count = 3`, it holds the indices of the three closest cities with their great-circle distances. The reduction operator chooses the three smallest distances from arguments `a` and `b`, each with three distances of their own. `combine_scores_t` is conscientiously designed to avoid dynamic indexing into any of the terms. We want to avoid spilling to high-latency local memory (dynamic indexing causes spilling into local memory), so we employ the compile-time loop-unwinding function `iterate<>`. We iteratively choose the smallest distance at the front of `a` or `b`, rotating forward the remaining distances of the respective argument. In theoretical terms this is an inefficient implementation, but rather than using memory operations it eats into register-register compute throughput which the GPU has in abundance.

```cpp
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
```
The score we use to rank city-city distances uses this great circle calculation. `earth_distance` is an ordinary math function tagged with `MGPU_HOST_DEVICE`, letting us call it from the `lbs_segreduce` device-tagged lambda. The power of C++11 and the design of moderngpu 2.0 let's us separate concerns: the numerics know nothing about the high-level transformer and the high-level transformer knows nothing about the numerics. It's the user's choice of lambda that brings them together into a single optimized kernel.

```cpp
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
```
`compute_distances` returns an array with `best_t<d>` structs for each city sorted in ascending distance of the (d-1'th) interval. This demo function uses several high-level transforms. `scan` and `interval_expand` are used together to turn the `cities_per_state` array (52 elements including District of Columbia and Puerto Rico) into a 29511-element map of cities intto the index of the first city in the same state, `city_to_city_map`. This provides a starting point inside the load-balancing search segmented reducer for computing the exact city-city distance pair of the work-item.

We utilize a brute-force all-pairs reducer. `transform_lbs` is used to record the number of distance calculations per city, `cities_per_state[state] - 1`, where the array element is loaded using tuple caching. The prefix sum of this generates `distance_segments`, the segments-descriptor array for the segmented reduction. We require two integers of storage per city. This is on the order of O(n) temporary memory, very storage-efficient compared to the O(n^2) number of distance calculations. 

The operative `lbs_segreduce` call requires only trivial storage space. Its usage here is as both a mapper and reducer: it expands work-items to compute all 25 million distance calculations, then reduces those results into 29 thousand closest-city tuples. GPUs are often memory capacity-limited devices, and the ability to perform on-the-fly expansions and reductions can be a make-or-break feature.

`lbs_segreduce` is a different, more fundamentally powerful function than the `transform_segreduce` used in [sparse matrix * vector](sparse-matrix--vector). The segment ID and rank-within-segment index are delivered to the user-provided lambda. In this query, the segment is the city and the rank is the index of the city within the state to compute the distance with. The rank is added to the iterator-cached `city_to_city_map` term to produce a global city index. The positions of each city are loaded (one from a cached iterator), the great-circle distance is computed, and a `best_t<>` structure is returned for reduction.

Finally, the moderngpu library function `segmented_sort_indices` sorts each struct of distances and indices by ascending order within each state. The most-connected cities occur first, the most-remote cities last. The demo source writes all 29 thousand cities to a file and prints the most-remote cities to the console.

By our measure [Attu Station, AK](https://en.wikipedia.org/wiki/Attu_Station,_Alaska) is the most remote city in the United States. It's 435 miles from the nearest census-designated place and 712 miles from the third-nearest place. At the western extent of the Aleutian islands, Attu Station is actually in the eastern hemisphere and much closer to Russia than to the continental United States.

This demo is a starting point for all sorts of geographic data sciences queries. Moving from census data to Twitter, Instagram or Facebook data, we can imagine huge volumes of geo-tagged data available for accelerated analytics. By using moderngpu's high-level transforms, the business intelligence is kept inside the user's reduction operators; the library solves the parallel decomposition challenges of the GPU.

# Examples of dynamic work creation
## Improved breadth-first search

Features demonstrated:

1. `lbs_workcreate`

#### `bfs2.cu`

```cpp
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
  int* out_edge_indices_data = edge_indices.data();
  mem_t<int> segments = engine.downsweep(
    [=]MGPU_DEVICE(int dest_seg, int index, int seg, int rank, tuple<int> desc) {
      // Return the same count as before and store output segment-specific
      // data using dest_index.
      int neighbor = edges[get<0>(desc) + rank];
      int begin = vertices[neighbor];
      int end = vertices[neighbor + 1];

      out_edge_indices_data[dest_seg] = begin;
      return end - begin;
    }, make_tuple(wl.edge_indices.data())
  );

  // Update the workload.
  wl.count = wl2_count.count;
  wl.num_segments = wl2_count.num_segments;
  wl.segments = std::move(segments);
  wl.edge_indices = std::move(edge_indices);
}
```
```
NUM VERTICES = 434102    NUM_EDGES = 32073440
Front for level 0 has 163 vertices and 21112 edges.
Front for level 1 has 165 vertices and 9561 edges.
Front for level 2 has 1067 vertices and 98990 edges.
Front for level 3 has 5961 vertices and 900032 edges.
Front for level 4 has 32678 vertices and 4617596 edges.
Front for level 5 has 131856 vertices and 15439710 edges.
Front for level 6 has 155163 vertices and 8686685 edges.
Front for level 7 has 71401 vertices and 1781177 edges.
Front for level 8 has 24208 vertices and 365748 edges.
Front for level 9 has 7670 vertices and 113306 edges.
Front for level 10 has 2324 vertices and 24539 edges.
Front for level 11 has 814 vertices and 7215 edges.
Front for level 12 has 410 vertices and 5198 edges.
Front for level 13 has 132 vertices and 1871 edges.
Front for level 14 has 52 vertices and 427 edges.
Front for level 15 has 7 vertices and 18 edges.
Front for level 16 has 6 vertices and 17 edges.
Front for level 17 has 4 vertices and 15 edges.
Front for level 18 has 4 vertices and 12 edges.
Front for level 19 has 1 vertices and 5 edges.
Front for level 20 has 3 vertices and 10 edges.
Front for level 21 has 2 vertices and 6 edges.
Front for level 22 has 3 vertices and 11 edges.
Front for level 23 has 6 vertices and 15 edges.
Front for level 24 has 1 vertices and 1 edges.
Front for level 25 has 0 vertices and 0 edges.
```
`bfs2.cu` is an improved breadth-first search. Like the [naive version](#breadth-first-search), each segment of work represents one vertex on the current level; each work-item represents a neighbor connected to the active-level vertex by an out-going edge. 

The naive version checked each vertex on each iteration to see if it was on the active front, and if it was, emitted its number of out-going edges at work-items. All of these counts were then scanned.

This improved version uses dynamic work creation to emit a request for new work (on the next round) when the CUDA intrinsic `atomicCAS` successfully sets the state of a vertex from unvisited to visited. If only 50 vertices are set to the visited state in a round, only 50 segments of work will be created, and no operations with costs that scale with the total number of vertices will be executed. 

Note that the `atomicCAS` call is only made during the `upsweep` phase. The actual count of work-items to emit is implicit in the number of out-going edges from that vertex; the `atomicCAS` really only determines if all those out-going edges materialize to work-items or not.

## Bit-compressed breadth-first search

Features demonstrated:

1. `lbs_workcreate`

#### `bfs3.cu`
```cpp
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
  int* out_edge_indices_data = edge_indices.data();
  mem_t<int> segments = engine.downsweep(
    [=]MGPU_DEVICE(int dest_seg, int index, int seg, int rank, 
      tuple<int> desc) {
      // Return the same count as before and store output segment-specific
      // data using dest_seg.
      int neighbor = edges[get<0>(desc) + rank];
      int begin = vertices[neighbor];
      int end = vertices[neighbor + 1];

      // Store the pointer into the edges array for the new work segment.
      out_edge_indices_data[dest_seg] = begin;

      // Stream out the vertex index.
      frontier_vertices[dest_seg] = neighbor;

      return end - begin;
    }, make_tuple(wl.edge_indices.data())
  );

  // Update the workload.
  wl.count = wl2_count.count;
  wl.num_segments = wl2_count.num_segments;
  wl.segments = std::move(segments);
  wl.edge_indices = std::move(edge_indices);

  return wl.num_segments;
}
```
The third demo version of breadth-first search uses the same work-creation mechanism of `bfs2.cu` but now utilizes the data streaming capability as well. The model for the second implementation was to store the level of each vertex (visited or not) in a contiguous integer array. On the work-creation _upsweep_ phase, an `atomicCAS` would attempt to store the next level in the search as the vertex's level. This would succeed only if the vertex hadn't already been visited. The downsweep phase would then request work for this newly visited vertex.

While this was an easy model, it was cache-inefficient. The slow part of top-down breadth-first search is the many disorganized atomic operations into an array representing vertex visitation. `bfs3.cu` improves on the earlier implementations by storing vertex visitation status as a single bit in a `visited_bits` array. The bit array is initially cleared. Before the BFS loop is started, each source vertex pokes in a bit to prevent itself from being visited from an incoming edge.

During _upsweep_ the bit corresponding to the vertex for each work-item is atomically set. If this was a successful operation, the upsweep lambda returns the number of outgoing edges of this newly visited vertex, just like `bfs2.cu`. The _downsweep_ lambda now streams the index of each newly visited vertex into a `frontier_vertices` array. On return, the caller advances the `frontier_vertices` pointer by the number of streamed-out vertices.

The performance advantage of this different implementation is that the region of memory undergoing heavy atomic operations is 32 times smaller, so we can expect much better utilization of the GPU's limited L2 cache. The convenience advantage is that the IDs of the connected vertices end up sorted by their distance from the source. 
