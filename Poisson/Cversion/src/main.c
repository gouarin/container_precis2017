#include <petscdmda.h>
#include <petscksp.h>

#include <domain.h>
#include <init.h>
#include <poisson.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv){
  PetscErrorCode ierr;
  int nx = 257, ny = 257;
  DM dm;
  PetscBool flg;
  Mat A;
  Vec u, b;
  KSP solver;
  PC pc;
  double norm;
  PetscInt stage;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nx", &nx, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-ny", &ny, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, PETSC_NULL, "-assemble", &flg);CHKERRQ(ierr);
 
  ierr = PetscLogStageRegister("Domain creation",&stage);
  ierr = PetscLogStagePush(stage);

  ierr = createDomain(&dm, nx, ny);CHKERRQ(ierr);

  ierr = PetscLogStagePop();

  ierr = PetscLogStageRegister("Matrix creation",&stage);
  ierr = PetscLogStagePush(stage);
  ierr = createMat(dm, &A, flg);CHKERRQ(ierr);
  ierr = PetscLogStagePop();
  
  ierr = PetscLogStageRegister("Second member creation",&stage);
  ierr = PetscLogStagePush(stage);
  ierr = DMCreateGlobalVector(dm, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(b, &u);CHKERRQ(ierr);
  ierr = init2d(dm, b);CHKERRQ(ierr);
  ierr = PetscLogStagePop();

  ierr = PetscLogStageRegister("KSP creation",&stage);
  ierr = PetscLogStagePush(stage);
  ierr = KSPCreate(PETSC_COMM_WORLD, &solver);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(solver, "poisson_");CHKERRQ(ierr);
  ierr = KSPSetOperators(solver, A, A);CHKERRQ(ierr);
  ierr = KSPSetType(solver, KSPCG);
  ierr = KSPGetPC(solver, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(solver);CHKERRQ(ierr);
  ierr = PetscLogStagePop();
  
  ierr = PetscLogStageRegister("Solver",&stage);
  ierr = PetscLogStagePush(stage);
  ierr = KSPSolve(solver, b, u);CHKERRQ(ierr);
  ierr = PetscLogStagePop();

  //ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  //sleep(10);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = KSPDestroy(&solver);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
