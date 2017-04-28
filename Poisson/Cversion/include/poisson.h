#ifndef POISSON_H
#define POISSON_H
#include <domain.h>

typedef struct MatContext{
  DM dm;
}MatContext;

// Finite difference
PetscErrorCode poissonFD2d_petsc(DM dm, Mat A);
PetscErrorCode poissonFD2dMatVec(Vec x, Vec y, DM dm);
PetscErrorCode PoissonMatMult2d(Mat A, Vec x, Vec y) ;
PetscErrorCode createMat(DM dm, Mat *A, PetscBool kind);

#endif
