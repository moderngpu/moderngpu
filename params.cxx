#include <moderngpu/meta.hxx>
#include <moderngpu/tuple.hxx>


using namespace mgpu;

int main(int argc, char** argv) {

  auto foo = make_tuple("hi", 19.1, 'f', 100);
  printf("%s %f %c %d\n", 
    get<0>(foo),
    get<1>(foo),
    get<2>(foo),
    get<3>(foo));

  auto&& restricted = restrict_tuple(foo);
  auto&& x0 = get<0>(foo);

  printf("%d\n", is_restrict<decltype(x0)>::value);
  
  printf("%d %d %d %d\n",
    is_restrict<decltype(get<0>(restricted))>::value,
    is_restrict<decltype(get<1>(restricted))>::value,
    is_restrict<decltype(get<2>(restricted))>::value,
    is_restrict<decltype(get<3>(restricted))>::value
  );

  /*
  typedef tuple<const int* __restrict__, int* __restrict__, char,
    char* __restrict__> tpl_t;
  tpl_t tpl = tpl_t();
  auto x0 = get<0>(tpl);
  auto x1 = get<1>(tpl);
  auto x2 = get<2>(tpl);
  auto x3 = get<3>(tpl);

  typename tuple_element<0, tpl_t>::type y0 = get<0>(tpl);
  typename tuple_element<1, tpl_t>::type y1 = get<1>(tpl);
  typename tuple_element<2, tpl_t>::type y2 = get<2>(tpl);
  typename tuple_element<3, tpl_t>::type y3 = get<3>(tpl);

  decltype(get<0>(tpl)) z0 = get<0>(tpl);
  decltype(get<1>(tpl)) z1 = get<1>(tpl);
  decltype(get<2>(tpl)) z2 = get<2>(tpl);
  decltype(get<3>(tpl)) z3 = get<3>(tpl);

  auto&& w0 = get<0>(tpl);
  auto&& w1 = get<1>(tpl);
  auto&& w2 = get<2>(tpl);
  auto&& w3 = get<3>(tpl);

  add_restrict<decltype(get<0>(tpl))>::type a0 = get<0>(tpl);
  add_restrict<decltype(get<1>(tpl))>::type a1 = get<1>(tpl);
  add_restrict<decltype(get<2>(tpl))>::type a2 = get<2>(tpl);
  add_restrict<decltype(get<3>(tpl))>::type a3 = get<3>(tpl);

  tuple<const char*, double*, int, float*> f;
  auto restricted = restrict_tuple(f);

  printf("%d %d %d %d\n", 
    is_restrict<decltype(x0)>::value,
    is_restrict<decltype(x1)>::value,
    is_restrict<decltype(x2)>::value,
    is_restrict<decltype(x3)>::value
  );

  printf("%d %d %d %d\n", 
    is_restrict<decltype(y0)>::value,
    is_restrict<decltype(y1)>::value,
    is_restrict<decltype(y2)>::value,
    is_restrict<decltype(y3)>::value
  );

  printf("%d %d %d %d\n", 
    is_restrict<decltype(z0)>::value,
    is_restrict<decltype(z1)>::value,
    is_restrict<decltype(z2)>::value,
    is_restrict<decltype(z3)>::value
  );

  printf("%d %d %d %d\n", 
    is_restrict<decltype(w0)>::value,
    is_restrict<decltype(w1)>::value,
    is_restrict<decltype(w2)>::value,
    is_restrict<decltype(w3)>::value
  );

  printf("%d %d %d %d\n", 
    is_restrict<decltype(a0)>::value,
    is_restrict<decltype(a1)>::value,
    is_restrict<decltype(a2)>::value,
    is_restrict<decltype(a3)>::value
  );*/
}

#if 0
#include <moderngpu/meta.hxx>
#include <moderngpu/operators.hxx>
#include <moderngpu/tuple.hxx>

using namespace mgpu;



int main(int argc, const char**) {


  auto foo = tuple_cat(tuple<int>(5), tuple<double, char>(1.2, 'f'),
    tuple<double>(19.1));


  auto foo2 = foo;
  get<2>(foo2)--;

  bool cmp = foo < foo2;
  
  auto f = [](int i, float f, const char* s) {
    printf("%d %f %s\n", i, f, s);
  };
  expand_tuple(f, make_tuple(19, 1.23f, "Hi there\n"));

  return 0;
}
#endif