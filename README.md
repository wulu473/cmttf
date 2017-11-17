[![Build Status](https://travis-ci.com/wulu473/cmttf.svg?token=CQhcTo9qCd3kF5dQ2izE&branch=master)](https://travis-ci.com/wulu473/cmttf)

Centre Manifold Theory based thin film simulations
==================================================

This is a software package to simulate thin film flows under shear stress, gravity and surface tension. Currently two systems are implemented. The first one is the classic thin film equation based on lubrication theory and dimensional analysis. The other one is based on centre manifold theory and takes also inertia into account. 

Platform specfic paths and dependencies
---------------------------------------

Platform specfic paths and dependencies are defined by creating a makefile in which `common.make` is included. See for example the file `lsc.make`. The system has to be specified passing the variable `SYSTEM` to the makefile. E.g.

```
make SYSTEM=roberts
```

Tests
-----

In order to run all tests use the `test` target in the makefile

### Unit-tests

Run unit-tests target in makefile

### Verification

Run regression-tests target in makefile

References
----------

1. Moriarty, J. a, Schwartz, L. W., & Tuck, E. O. (1991). Unsteady spreading of thin liquid films with small surface tension. Physics Of Fluids A Fluid Dynamics, 3(5), 733. http://doi.org/10.1063/1.858006
2. Roberts, A. J., & Li, Z. (2006). An accurate and comprehensive model of thin fluid flows with inertia on curved substrates. Journal of Fluid Mechanics, 553, 33. http://doi.org/10.1017/S0022112006008640

