#include "petsc.h"
#include <init.h>

PetscErrorCode init2d(DM dm, Vec b){
  PetscErrorCode ierr;
  double x, y;
  int i, j;
  DMDALocalInfo info;
  double hx, hy;
  PetscScalar **pb;

  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(dm, b, &pb);CHKERRQ(ierr);
  
  hx = 1./(info.mx-1);
  hy = 1./(info.mx-1);
  
  for(i=info.xs; i<info.xs+info.xm; i++)
    for(j=info.ys; j<info.ys+info.ym; j++){
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1)
        pb[j][i] = 0.;
      else
        pb[j][i] = 1.;
    }

  ierr = DMDAVecRestoreArray(dm, b, &pb);CHKERRQ(ierr);
}
