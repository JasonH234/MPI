/*
** code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the bhatnagar-gross-krook collision step.
**
** the 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** a 2d grid:
**
**           cols
**       --- --- ---
**      | d | e | f |
** rows  --- --- ---
**      | a | b | c |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1d array:
**
**  --- --- --- --- --- ---
** | a | b | c | d | e | f |
**  --- --- --- --- --- ---
**
** grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stddef.h>
#include "lbm.h"
#include "mpi.h"

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char * final_state_file = NULL;
    char * av_vels_file = NULL;
    char * param_file = NULL;

    accel_area_t accel_area;

    param_t  params;              /* struct to hold parameter values */
    speed_t* cells = NULL;
    speed_t* cells_even = (speed_t*) malloc(sizeof(speed_t)*(params.ny*params.nx));    /* grid containing fluid densities */
    speed_t* cells_odd = (speed_t*) malloc(sizeof(speed_t)*(params.ny*params.nx));    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */

    int    ii;                    /*  generic counter */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
    double usrtim;                /* floating point number to record elapsed user CPU time */
    double systim;                /* floating point number to record elapsed system CPU time */

    int flag, size, rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(NULL, NULL);
    MPI_Initialized(&flag);
    if(flag == 0) {
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //Name
    MPI_Get_processor_name(processor_name, &name_len);
    //Process number
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //Rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
      const int count = 8;
      int blocks[8] = {1,1,1,1,1,1,1,1};
      MPI_Aint offsets[8];
      offsets[0] = offsetof(param_t, nx);
      offsets[1] = offsetof(param_t, ny);
      offsets[2] = offsetof(param_t, max_iters);
      offsets[3] = offsetof(param_t, tot_cells);
      offsets[4] = offsetof(param_t, reynolds_dim);
      offsets[5] = offsetof(param_t, density);
      offsets[6] = offsetof(param_t, accel);
      offsets[7] = offsetof(param_t, omega);
      MPI_Datatype types[8] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
			       MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
      MPI_Datatype MPI_PARAM_T;

      MPI_Type_create_struct(count, blocks, offsets, types, &MPI_PARAM_T);
      MPI_Type_commit(&MPI_PARAM_T);

    // master initialise
    if(rank == 0) {
      parse_args(argc, argv, &final_state_file, &av_vels_file, &param_file);
      initialise(param_file, &accel_area, &params, &cells_even, &cells_odd, &obstacles, &av_vels);
      
      gettimeofday(&timstr,NULL);
      tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    }
    // share params
    MPI_Bcast(&params, 1, MPI_PARAM_T, 0, MPI_COMM_WORLD);
    
    
   
    printf("Hello world from processor %s, rank %d, out of %d processors\n", processor_name, rank, size);
    printf("I have %d,%d total cell grid size. Omg %f\n", params.nx, params.ny, params.omega);
    int expected_cells = (rank == size-1) ? (params.ny%size + params.ny/size) * params.nx : (params.ny/size) * params.nx;
    printf("Expecting %d cells\n", expected_cells);

    for (ii = 0; ii < params.max_iters; ii++)
    {
      /*	if(ii % 2 == 0) {
	  accelerate_flow(params, accel_area, cells_even, obstacles);
	  av_vels[ii] = simulation_steps(params, cells_odd, cells_even, obstacles);
	}
	else {
	  accelerate_flow(params, accel_area, cells_odd, obstacles);
	  av_vels[ii] = simulation_steps(params, cells_even, cells_odd, obstacles);  
	}
      */
        #ifdef DEBUG
        printf("==timestep: %d==\n", ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n", total_density(params, cells_even));
        #endif
    }

    
    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr=ru.ru_utime;
    usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    timstr=ru.ru_stime;
    systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    if(rank == 0) {
      printf("Process %d ==done==\n", rank);
      printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params,av_vels[ii-1]));
      printf("Elapsed time:\t\t\t%.6f (s)\n", toc-tic);
      printf("Elapsed user CPU time:\t\t%.6f (s)\n", usrtim);
      printf("Elapsed system CPU time:\t%.6f (s)\n", systim);

      write_values(final_state_file, av_vels_file, params, cells_even, obstacles, av_vels);
      finalise(&cells_even, &cells_odd, &obstacles, &av_vels);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}

void write_values(const char * final_state_file, const char * av_vels_file,
    const param_t params, speed_t* cells, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    int ii,jj,kk;                 /* generic counters */
    const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(final_state_file, "w");

    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
            /* an occupied cell */
            if (obstacles[ii*params.nx + jj])
            {
                u_x = u_y = u = 0.0;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii*params.nx + jj].speeds[kk];
                }

                /* compute x velocity component */
                u_x = (cells[ii*params.nx + jj].speeds[1] +
                        cells[ii*params.nx + jj].speeds[5] +
                        cells[ii*params.nx + jj].speeds[8]
                    - (cells[ii*params.nx + jj].speeds[3] +
                        cells[ii*params.nx + jj].speeds[6] +
                        cells[ii*params.nx + jj].speeds[7]))
                    / local_density;

                /* compute y velocity component */
                u_y = (cells[ii*params.nx + jj].speeds[2] +
                        cells[ii*params.nx + jj].speeds[5] +
                        cells[ii*params.nx + jj].speeds[6]
                    - (cells[ii*params.nx + jj].speeds[4] +
                        cells[ii*params.nx + jj].speeds[7] +
                        cells[ii*params.nx + jj].speeds[8]))
                    / local_density;

                /* compute norm of velocity */
                u = sqrt((u_x * u_x) + (u_y * u_y));

                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",
                jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
        }
    }

    fclose(fp);

    fp = fopen(av_vels_file, "w");
    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

    for (ii = 0; ii < params.max_iters; ii++)
    {
        fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);
}

float calc_reynolds(const param_t params, const float av_vel)
{
    const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

    return av_vel * params.reynolds_dim / viscosity;
}

float total_density(const param_t params, speed_t* cells)
{
    int ii,jj,kk;        /* generic counters */
    float total = 0.0;  /* accumulator */

    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.ny; jj++)
        {
            for (kk = 0; kk < NSPEEDS; kk++)
            {
                total += cells[ii*params.nx + jj].speeds[kk];
            }
        }
    }

    return total;
}

