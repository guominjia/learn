# numpy

## [umathmodule.c](https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/umathmodule.c)
`initumath` set `pi, e, etc`  
`initumath` invoke `InitOperator` to set `add, sub, etc`

## [generate_umath.c](https://github.com/numpy/numpy/blob/main/numpy/_core/code_generators/generate_umath.py)
`InitOperator` set `add, sub, etc`

## Deep dived issue

- [Data race](https://github.com/numpy/numpy/issues/30085)

## References
- https://github.com/numpy/numpy