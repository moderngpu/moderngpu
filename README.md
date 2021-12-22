# moderngpu
moderngpu is a productivity library for general-purpose computing on GPUs. It is a header-only C++ library written for CUDA. The unique value of the library is in its accelerated primitives for solving irregularly parallel problems. 

- **(c) 2021 [Sean Baxter](http://twitter.com/seanbax)** 
- **You can drop me a line [here](mailto:moderngpu@gmail.com)**
- Full documentation with [github wiki](https://github.com/moderngpu/moderngpu/wiki) under heavy construction.

## Quick Start Guide
```bash
git clone https://github.com/moderngpu/moderngpu.git
cd moderngpu
mkdir build && cd build
cmake ..
make # or make name_of_project to build a specific binary
./bin/test_segreduce
```

## How to Cite

```
@Unpublished{     Baxter:2016:M2,
  author        = {Baxter, Sean},
  title         = {moderngpu 2.0},
  note          = {\url{https://github.com/moderngpu/moderngpu/wiki}},
  year          = 2016
}
```
