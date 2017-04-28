#include <petscvec.h>
#include <petscmat.h>
#include <petscdmda.h>
#include <poisson.h>

#undef __FUNCT__
#define __FUNCT__ "createMat"
PetscErrorCode createMat(DM dm, Mat *A, PetscBool kind){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MatContext *matctx;
  DMDALocalInfo info;
  int localsize, totalsize;

  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);

  if (kind){
    ierr = DMCreateMatrix(dm, A);CHKERRQ(ierr);
    ierr = poissonFD2d_petsc(dm, *A);CHKERRQ(ierr);
  }
  else{
    matctx = malloc(sizeof(MatContext));
    ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
    
    localsize = info.xm*info.ym*info.zm;
    totalsize = info.mx*info.my*info.mz;
    matctx->dm = dm;

    ierr = MatCreateShell(PETSC_COMM_WORLD, localsize, localsize, totalsize, totalsize, (void*) matctx, A);CHKERRQ(ierr);

    ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))PoissonMatMult2d);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PoissonMatMult2d"
PetscErrorCode PoissonMatMult2d(Mat A, Vec x, Vec y) 
{
  PetscFunctionBegin;
  MatContext *ctx;
  PetscErrorCode ierr;
  DMDALocalInfo info;

  ierr = MatShellGetContext(A, (void**) &ctx);CHKERRQ(ierr);
  ierr = poissonFD2dMatVec(x, y, ctx->dm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "poissonFD2dMatVec"
PetscErrorCode poissonFD2dMatVec(Vec x, Vec y, DM dm){
  PetscErrorCode ierr;
  PetscFunctionBegin;

  double hx, hy;
  int i, j;
  double cx, cy, cd;

  DMDALocalInfo info;

  Vec xLocal, yLocal;
  PetscScalar **px, **py;

  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  hx = 1./(info.mx-1);
  hy = 1./(info.my-1);
  cx=-1./(hx*hx), cy=-1./(hy*hy), cd=2./(hx*hx)+2./(hy*hy);
 
  ierr = DMGetLocalVector(dm, &xLocal);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm, x, INSERT_VALUES, xLocal);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dm, x, INSERT_VALUES, xLocal);CHKERRQ(ierr); 
  
  ierr = DMGetLocalVector(dm, &yLocal);CHKERRQ(ierr);
  ierr = VecSet(yLocal, 0.);CHKERRQ(ierr); 
  ierr = VecSet(y, 0.);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(dm, xLocal, &px);CHKERRQ(ierr); 
  ierr = DMDAVecGetArray(dm, yLocal, &py);CHKERRQ(ierr); 

  for(j=info.ys; j<info.ys+info.ym; j++)
    for(i=info.xs; i<info.xs+info.xm ; i++){
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1){
        // Dirichlet condition
        py[j][i] = px[j][i];
      }
      else{
        // interior point
        py[j][i] = cd*px[j][i] + cx*(px[j][i+1] + px[j][i-1]) + cy*(px[j+1][i] + px[j-1][i]);
      }
    }
  
  ierr = DMDAVecRestoreArray(dm, xLocal, &px);CHKERRQ(ierr); 
  ierr = DMDAVecRestoreArray(dm, yLocal, &py);CHKERRQ(ierr); 
  ierr = DMRestoreLocalVector(dm, &xLocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &yLocal);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm, yLocal, ADD_VALUES, y);CHKERRQ(ierr); 
  ierr = DMLocalToGlobalEnd(dm, yLocal, ADD_VALUES, y);CHKERRQ(ierr); 
  ierr = PetscLogFlops(7.*info.mx*info.my);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "poissonFD2d_petsc"
PetscErrorCode poissonFD2d_petsc(DM dm, Mat A)
{  
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMDALocalInfo info;

  int i,j;
  MatStencil row, col[5];
  PetscScalar coef, coef5[5];
  
  DMDAGetLocalInfo(dm, &info);

  double hx2;
  double hy2;

  hx2 = 1./((info.mx-1)*(info.mx-1));
  hy2 = 1./((info.my-1)*(info.my-1));

  coef = 1.;
  coef5[0] = 2./hx2 + 2./hy2; 
  coef5[1] = -1./hx2; coef5[2] = -1./hx2;
  coef5[3] = -1./hy2; coef5[4] = -1./hy2;

  for(j=info.ys; j<info.ys+info.ym; j++)
    for(i=info.xs; i<info.xs+info.xm ; i++){
      row.i = i; row.j = j; row.c = 0;
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1){
        // Dirichlet condition
        MatSetValuesStencil(A, 1, &row, 1, &row, &coef, INSERT_VALUES);
      }
      else{
        // interior point
        col[0].i = i; col[0].j = j; col[0].c = 0;
        col[1].i = i-1; col[1].j = j; col[1].c = 0;
        col[2].i = i+1; col[2].j = j; col[2].c = 0;
        col[3].i = i; col[3].j = j-1; col[3].c = 0;
        col[4].i = i; col[4].j = j+1; col[4].c = 0;
        MatSetValuesStencil(A, 1, &row, 5, col, coef5, INSERT_VALUES);
      }
    }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

