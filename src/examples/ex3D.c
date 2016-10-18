/*
   Example 3D Laplace

   Interface:      Structured interface (Struct)

   Compile with:   make ex4

   Sample run:     mpirun -np 16 ex4 -n 33 -solver 10 -K 3 -B 0 -C 1 -U0 2 -F 4

   To see options: ex4 -help

   Description:    This example differs from the previous structured example
                   (Example 3) in that a more sophisticated stencil and
                   boundary conditions are implemented. The method illustrated
                   here to implement the boundary conditions is much more general
                   than that in the previous example.  Also symmetric storage is
                   utilized when applicable.

                   This code solves the convection-reaction-diffusion problem
                   div (-K grad u + B u) + C u = F in the unit square with
                   boundary condition u = U0.  The domain is split into N x N
                   processor grid.  Thus, the given number of processors should
                   be a perfect square. Each processor has a n x n grid, with
                   nodes connected by a 5-point stencil. Note that the struct
                   interface assumes a cell-centered grid, and, therefore, the
                   nodes are not shared.

                   To incorporate the boundary conditions, we do the following:
                   Let x_i and x_b be the interior and boundary parts of the
                   solution vector x. If we split the matrix A as
                             A = [A_ii A_ib; A_bi A_bb],
                   then we solve
                             [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
                   Note that this differs from the previous example in that we
                   are actually solving for the boundary conditions (so they
                   may not be exact as in ex3, where we only solved for the
                   interior).  This approach is useful for more general types
                   of b.c.

                   A number of solvers are available. More information can be
                   found in the Solvers and Preconditioners chapter of the
                   User's Manual.

                   We recommend viewing examples 1, 2, and 3 before viewing this
                   example.
*/

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE_struct_ls.h"

#ifdef M_PI
  #define PI M_PI
#else
  #define PI 3.14159265358979
#endif

#include "vis.c"

/* Macro to evaluate a function F in the grid point (i,j) */
#define Eval(F,i,j,k) (F( (ilower[0]+(i))*h, (ilower[1]+(j))*h, (ilower[2]+(k))*h ))
#define bcEval(F,i,j,k) (F( (bc_ilower[0]+(i))*h, (bc_ilower[1]+(j))*h, (bc_ilower[2]+(k))*h ))

int optionK;

/* Diffusion coefficient */
double K(double x, double y, double z)
{
   switch (optionK)
   {
      case 0:
         return 1.0;
      case 1:
         return x*x+exp(y);
      case 2:
         if ((fabs(x-0.5) < 0.25) && (fabs(y-0.5) < 0.25))
            return 100.0;
         else
            return 1.0;
      case 3:
         if (((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)) < 0.0625)
            return 10.0;
         else
            return 1.0;
      default:
         return 1.0;
   }
}

/* Reaction coefficient */
double C(double x, double y, double z)
{
    return 0.1;
}

/* Boundary condition */
double U0(double x, double y, double z)
{
    return 0.0;
}

/* Right-hand side */
double F(double x, double y, double z)
{
    return 2*PI*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z);
}

int main (int argc, char *argv[])
{
   int i, j, k;
   int it;

   int myid, num_procs;

   int n, N, pi, pj, pk;
   double h, h3;
   int ilower[3], iupper[3];

   int n_pre, n_post;
   int rap, relax, skip;
   int time_index;

   int num_iterations;
   double final_res_norm;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructSolver   solver;
   HYPRE_StructSolver   precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Set default parameters */
   n         = 33;
   optionK   = 0;
   n_pre     = 1;
   n_post    = 1;
   rap       = 0;
   relax     = 1;
   skip      = 0;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-K") == 0 )
         {
            arg_index++;
            optionK = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-v") == 0 )
         {
            arg_index++;
            n_pre = atoi(argv[arg_index++]);
            n_post = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rap") == 0 )
         {
            arg_index++;
            rap = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-relax") == 0 )
         {
            arg_index++;
            relax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-skip") == 0 )
         {
            arg_index++;
            skip = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -n  <n>             : problem size per processor (default: 8)\n");
         printf("  -K  <K>             : choice for the diffusion coefficient (default: 1)\n");
         printf("  -v <n_pre> <n_post> : number of pre and post relaxations\n");
         printf("  -rap <r>            : coarse grid operator type\n");
         printf("                        0 - Galerkin (default)\n");
         printf("                        1 - non-Galerkin ParFlow operators\n");
         printf("                        2 - Galerkin, general operators\n");
         printf("  -relax <r>          : relaxation type\n");
         printf("                        0 - Jacobi\n");
         printf("                        1 - Weighted Jacobi (default)\n");
         printf("                        2 - R/B Gauss-Seidel\n");
         printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
         printf("  -skip <s>           : skip levels in PFMG (0 or 1)\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (N x N x N).  The local
      problem size is indicated by n (n x n x n). pi and pj
      indicate position in the processor grid. */
   N  = pow(num_procs,1.0/3.);
   h  = 1.0 / (N*n-1);
   h3 = h*h*h;
   pk = myid / N / N;
   pj = (myid-pk*N*N) / N;
   pi = myid - pj*N - pk*N*N;

   /* Define the nodes owned by the current processor (each processor's
      piece of the global grid) */
   ilower[0] = pi*n;
   ilower[1] = pj*n;
   ilower[2] = pk*n;
   iupper[0] = ilower[0] + n-1;
   iupper[1] = ilower[1] + n-1;
   iupper[2] = ilower[2] + n-1;

   /* 1. Set up a grid */
   {
      /* Create an empty 3D grid object */
      HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);

      /* Add a new box to the grid */
      HYPRE_StructGridSetExtents(grid, ilower, iupper);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
      /* Define the geometry of the stencil */
      int offsets[7][3] = {{0,0,0}, {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0},
        {0,0,-1}, {0,0,1}};

      /* Create an empty 3D, 7-pt stencil object */
      HYPRE_StructStencilCreate(3, 7, &stencil);

      /* Assign stencil entries */
      for (i = 0; i < 7; i++)
         HYPRE_StructStencilSetElement(stencil, i, offsets[i]);

   /* 3. Set up Struct Vectors for b and x */
   {
      double *values;

      /* Create an empty vector object */
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);

      values = (double*) calloc((n*n), sizeof(double));

      /* Set the values of b in left-to-right, bottom-to-top order */
      it = 0;
      for (k = 0; k < n; k++)
        for (j = 0; j < n; j++)
            for (i = 0; i < n; i++, it++)
                values[it] = h3 * Eval(F,i,j,k);
      HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

      /* Set x = 0 */
      for (i = 0; i < (n*n*n); i ++)
         values[i] = 0.0;
      HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

      free(values);

      /* Assembling is postponed since the vectors will be further modified */
   }

   /* 4. Set up a Struct Matrix */
   {
      /* Create an empty matrix object */
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Use symmetric storage? */
      HYPRE_StructMatrixSetSymmetric(A, 0);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_StructMatrixInitialize(A);

      /* Set the stencil values in the interior. Here we set the values
         at every node. We will modify the boundary nodes later. */
         int stencil_indices[7] = {0, 1, 2, 3, 4, 5, 6}; /* labels correspond
                                                      to the offsets */
         double *values;

         values = (double*) calloc(7*(n*n*n), sizeof(double));

         /* The order is left-to-right, bottom-to-top */
         it = 0;
         for (k = 0; k < n; k++)
         for (j = 0; j < n; j++)
            for (i = 0; i < n; i++, it+=7)
            {
               values[it+1] = - Eval(K,i-0.5,j,k);

               values[it+2] = - Eval(K,i+0.5,j,k);

               values[it+3] = - Eval(K,i,j-0.5,k);

               values[it+4] = - Eval(K,i,j+0.5,k);

               values[it+5] = - Eval(K,i,j,k-0.5);

               values[it+6] = - Eval(K,i,j,k+0.5);

               values[it] = Eval(C,i,j,k)
                  + Eval(K ,i-0.5,j,k) + Eval(K ,i+0.5,j,k)
                  + Eval(K ,i,j-0.5,k) + Eval(K ,i,j+0.5,k)
                  + Eval(K ,i,j,k-0.5) + Eval(K ,i,j,k+0.5);
            }

         HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 7,
                                        stencil_indices, values);

         free(values);
   }

   /* 5. Set the boundary conditions, while eliminating the coefficients
         reaching ouside of the domain boundary. We must modify the matrix
         stencil and the corresponding rhs entries. */
   {
      int bc_ilower[3];
      int bc_iupper[3];

      int stencil_indices[7] = {0, 1, 2, 3, 4, 5, 6};
      double *values, *bvalues;

      int nentries = 7;

      values  = (double*) calloc(nentries*n*n, sizeof(double));
      bvalues = (double*) calloc(n*n, sizeof(double));

      /* The stencil at the boundary nodes is 1-0-0-0-0. Because
         we have I x_b = u_0; */
      for (i = 0; i < nentries*n*n; i += nentries)
      {
         values[i] = 1.0;
         for (j = 1; j < nentries; j++)
            values[i+j] = 0.0;
      }

      /* Processors at y = 0 */
      if (pk == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;
         bc_ilower[2] = pk*n;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1] + n-1;
         bc_iupper[2] = bc_ilower[2];

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         it = 0;
         for (j = 0; j < n; j++)
         for (i = 0; i < n; i++, it++)
            bvalues[it] = bcEval(U0,i,j,0);

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at z = 1 */
      if (pk == N-1)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;
         bc_ilower[2] = pk*n + n-1;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1] + n-1;
         bc_iupper[2] = bc_ilower[2];

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         it = 0;
         for (j = 0; j < n; j++)
         for (i = 0; i < n; i++,it++)
            bvalues[it] = bcEval(U0,i,j,0);

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at y = 0 */
      if (pj == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;
         bc_ilower[2] = pk*n;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];
         bc_iupper[2] = bc_ilower[2] + n-1;

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         it = 0;
         for (k = 0; k < n; k++)
         for (i = 0; i < n; i++,it++)
            bvalues[it] = bcEval(U0,i,0,k);

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at y = 1 */
      if (pj == N-1)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n + n-1;
         bc_ilower[2] = pk*n;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];
         bc_iupper[2] = bc_ilower[2] + n-1;

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         it = 0;
         for (k = 0; k < n; k++)
         for (i = 0; i < n; i++,it++)
            bvalues[it] = bcEval(U0,i,0,k);

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at x = 0 */
      if (pi == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;
         bc_ilower[2] = pk*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;
         bc_iupper[2] = bc_ilower[2] + n-1;

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         it = 0;
         for (k = 0; k < n; k++)
         for (j = 0; j < n; j++,it++)
            bvalues[it] = bcEval(U0,0,j,k);

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at x = 1 */
      if (pi == N-1)
      {
         bc_ilower[0] = pi*n + n-1;
         bc_ilower[1] = pj*n;
         bc_ilower[2] = pk*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;
         bc_iupper[2] = bc_ilower[2] + n-1;

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         it = 0;
         for (k = 0; k < n; k++)
         for (j = 0; j < n; j++,it++)
            bvalues[it] = bcEval(U0,0,j,k);

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      free(values);
      free(bvalues);
   }

   /* Finalize the vector and matrix assembly */
   HYPRE_StructMatrixAssemble(A);
   HYPRE_StructVectorAssemble(b);
   HYPRE_StructVectorAssemble(x);

   /* 6. Set up and use a solver */
   /* Preconditioned GMRES */
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Note that GMRES can be used with all the interfaces - not
         just the struct.  So here we demonstrate the
         more generic GMRES interface functions. Since we have chosen
         a struct solver then we must type cast to the more generic
         HYPRE_Solver when setting options with these generic functions.
         Note that one could declare the solver to be
         type HYPRE_Solver, and then the casting would not be necessary.*/

      HYPRE_GMRESSetMaxIter((HYPRE_Solver) solver, 500 );
      HYPRE_GMRESSetKDim((HYPRE_Solver) solver,30);
      HYPRE_GMRESSetTol((HYPRE_Solver) solver, 1.0e-06 );
      HYPRE_GMRESSetPrintLevel((HYPRE_Solver) solver, 2 );
      HYPRE_GMRESSetLogging((HYPRE_Solver) solver, 1 );

         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetPrintLevel(precond, 0);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_StructGMRESSetPrecond(solver,
                                     HYPRE_StructSMGSolve,
                                     HYPRE_StructSMGSetup,
                                     precond);

      /* GMRES Setup */
      HYPRE_StructGMRESSetup(solver, A, b, x );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      /* GMRES Solve */
      HYPRE_StructGMRESSolve(solver, A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Get info and release memory */
      HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructGMRESDestroy(solver);

      HYPRE_StructSMGDestroy(precond);
   }

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

   /* Free memory */
   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
