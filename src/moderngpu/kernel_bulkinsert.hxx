// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "kernel_merge.hxx"

BEGIN_MGPU_NAMESPACE

// Insert the values at a_keys before the values at b_keys identified by
// insert.
template<typename launch_t = empty_t, typename a_it, typename insert_it, 
  typename b_it, typename c_it>
void bulk_insert(a_it a, insert_it a_insert, int insert_size, b_it b, 
  int source_size, c_it c, context_t& context) {

  merge<launch_t>(a_insert, a, insert_size, counting_iterator_t<int>(0), b, 
    source_size, discard_iterator_t<int>(), c, mgpu::less_t<int>(), context);
}

END_MGPU_NAMESPACE
