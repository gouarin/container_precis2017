#include <petscdmda.h>
#include <petscksp.h>
#include <domain.h>

#undef __FUNCT__
#define __FUNCT__ "createDomain"
PetscErrorCode createDomain(DM *dm, int nx, int ny){
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 
                      DMDA_STENCIL_STAR,
                      nx, ny, PETSC_DECIDE, PETSC_DECIDE,
                      1, 1, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
