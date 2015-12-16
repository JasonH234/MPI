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
    speed_t* cells_whole = NULL;
    speed_t* cells_even = NULL;    /* grid containing fluid densities */
    speed_t* cells_odd = NULL;
    int*     obstacles_whole = NULL;    /* grid indicating which cells are blocked */
    int* obstacles = NULL;
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

      const int count = 12;
      int blocks[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
      MPI_Aint offsets[12];
      offsets[0] = offsetof(param_t, nx);
      offsets[1] = offsetof(param_t, ny);
      offsets[2] = offsetof(param_t, max_iters);
      offsets[3] = offsetof(param_t, tot_cells);
      offsets[4] = offsetof(param_t, reynolds_dim);
      offsets[5] = offsetof(param_t, minX);
      offsets[6] = offsetof(param_t, maxX);
      offsets[7] = offsetof(param_t, minY);
      offsets[8] = offsetof(param_t, maxY);
      offsets[9] = offsetof(param_t, density);
      offsets[10] = offsetof(param_t, accel);
      offsets[11] = offsetof(param_t, omega);
      MPI_Datatype types[12] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                               MPI_INT, MPI_INT, MPI_INT, MPI_INT,
			                    MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
      MPI_Datatype MPI_PARAM_T;
      MPI_Type_create_struct(count, blocks, offsets, types, &MPI_PARAM_T);
      MPI_Type_commit(&MPI_PARAM_T);

      const int cell_count = 1;
      int cell_blocks[1] = {NSPEEDS};
      MPI_Aint cell_offsets[1] ={offsetof(speed_t, speeds)};
      MPI_Datatype cell_types[1] = {MPI_FLOAT};
      MPI_Datatype MPI_SPEED_T;
      MPI_Type_create_struct(cell_count, cell_blocks, cell_offsets, cell_types, &MPI_SPEED_T);
      MPI_Type_commit(&MPI_SPEED_T);

      // accel variables
    int idx;
    int is_row;

    // master initialise
    if(rank == 0) {
      parse_args(argc, argv, &final_state_file, &av_vels_file, &param_file);
      initialise(param_file, &accel_area, &params, &cells_whole, &obstacles_whole, &av_vels);
      
      //set accel variables for broadcast
      idx = accel_area.idx;
      if(accel_area.col_or_row == ACCEL_ROW)
	    is_row = 1;
      else
	    is_row = 0;

    }
    // share params
    MPI_Bcast(&params, 1, MPI_PARAM_T, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&is_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    const int croppedY = params.maxY - params.minY;
    const char sendUp = ((rank == 0) && croppedY < params.ny) ? '0' : '1';
    const char sendDown = ((rank == size-1) && croppedY < params.ny) ? '0' : '1';

    if(rank == 0)
        initialise_unused(params, &cells_whole);    

    const int group_size = ((int)croppedY/size) * params.nx;
    const int remainderRows = (croppedY%size);
    const int remainder = (croppedY%size)*params.nx;
    const int expected_cells = (rank < remainderRows) ? (params.nx + group_size) : group_size;
    const int padding = 2 * params.nx;
    //printf("%d, %d, %d\n", croppedY, (croppedY%size), expected_cells);
    if(rank > 0) {
      accel_area.idx = idx;
      accel_area.col_or_row = (is_row == 1) ? ACCEL_ROW : ACCEL_COLUMN;
    }

    int do_accel = 0;
    if(accel_area.col_or_row == ACCEL_ROW) {
      int low = (rank < remainderRows) ? rank * (((int) croppedY/size)+1) : rank * ((int) croppedY/size) + remainderRows;
      int high = (rank < remainderRows) ? (rank+1) * (((int) croppedY/size)+1) : (rank+1) * ((int) croppedY/size) + remainderRows;
      accel_area.idx -= params.minY;
      if(accel_area.idx >= low && accel_area.idx < high) {
	    do_accel = 1;
	    accel_area.idx -=low;
      }
    }

    // save size of full grid before setting to cropped size
    const int full_y = params.ny;
    params.ny = (rank < remainderRows) ? 1 + ((int) croppedY/size) : ((int) croppedY/size);

    int group_sizes[size];
    int displacements[size];
    for (ii = 0; ii < size; ii++) {
	    group_sizes[ii] = group_size;
	    displacements[ii] = params.minY*params.nx +(group_size) * (ii); 

        if(ii < remainderRows)  {
            group_sizes[ii] += params.nx;
            displacements[ii] += ii*params.nx;
        }
        else 
            displacements[ii] += remainderRows*params.nx;
    }
    // allocate and initialise cells memory, with two rows padding
    initialise_worker(params, &cells_even, &cells_odd, &obstacles, expected_cells);

    MPI_Scatterv(obstacles_whole, group_sizes, displacements, MPI_INT, obstacles, expected_cells, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    //Calculate offsets
    //const int start = ((rank == 0) && croppedY < full_y) ? expected_cells+params.nx : 0;
    const int end = expected_cells - params.nx;//((rank == size-1) && croppedY < full_y) ? expected_cells : expected_cells - params.nx;
    const int pad1 = expected_cells;
    const int pad2 = expected_cells + params.nx;
    //Calculate neighbour workers
    const int down = (rank+1)%size;
    const int up = (rank == 0) ? size-1 : rank-1; 

    if(rank == 0) {
        gettimeofday(&timstr,NULL);
        tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    }

    for (ii = 0; ii < params.max_iters; ii++)
    {
        float av_vel;
        if(ii % 2 == 0) {

	      if(do_accel)
	         accelerate_flow(params, accel_area, cells_even, obstacles);

	      // only even send
	      if(rank%2 == 0) {
            MPI_Send(&cells_even[0], params.nx, MPI_SPEED_T, 
	             up, 0, MPI_COMM_WORLD);
            MPI_Recv(&cells_even[pad2], params.nx, MPI_SPEED_T, up, 0, 
	             MPI_COMM_WORLD, NULL);
            MPI_Send(&cells_even[end], params.nx, MPI_SPEED_T, 
	             down, 0, MPI_COMM_WORLD);
            MPI_Recv(&cells_even[pad1], params.nx, MPI_SPEED_T, down, 0, 
	             MPI_COMM_WORLD, NULL);
	      }
	      else {
            MPI_Recv(&cells_even[pad1], params.nx, MPI_SPEED_T, down, 0, 
	             MPI_COMM_WORLD, NULL);
	        MPI_Send(&cells_even[end], params.nx, MPI_SPEED_T, 
		        down, 0, MPI_COMM_WORLD);
	        MPI_Recv(&cells_even[pad2], params.nx, MPI_SPEED_T, up, 0, 
		         MPI_COMM_WORLD, NULL);
	        MPI_Send(&cells_even[0], params.nx, MPI_SPEED_T, 
		         up, 0, MPI_COMM_WORLD);
	      }

	      av_vel = simulation_steps(params, cells_odd, cells_even, obstacles); 
	    }
	    else {
	      if(do_accel)
	        accelerate_flow(params, accel_area, cells_odd, obstacles);
	      
	      if(rank%2 == 0) {
	        MPI_Send(&cells_odd[0], params.nx, MPI_SPEED_T, 
		         up, 0, MPI_COMM_WORLD);
	        MPI_Recv(&cells_odd[pad2], params.nx, MPI_SPEED_T, up, 0, 
		         MPI_COMM_WORLD, NULL);
	        MPI_Send(&cells_odd[end], params.nx, MPI_SPEED_T, 
		         down, 0, MPI_COMM_WORLD);
	        MPI_Recv(&cells_odd[pad1], params.nx, MPI_SPEED_T, down, 0, 
		         MPI_COMM_WORLD, NULL);
	      }
	      else {
	        MPI_Recv(&cells_odd[pad1], params.nx, MPI_SPEED_T, down, 0, 
		         MPI_COMM_WORLD, NULL);
	        MPI_Send(&cells_odd[end], params.nx, MPI_SPEED_T, 
		         down, 0, MPI_COMM_WORLD);
	        MPI_Recv(&cells_odd[pad2], params.nx, MPI_SPEED_T, up, 0, 
		         MPI_COMM_WORLD, NULL);
	        MPI_Send(&cells_odd[0], params.nx, MPI_SPEED_T, 
		         up, 0, MPI_COMM_WORLD);
	      }
	      av_vel = simulation_steps(params, cells_even, cells_odd, obstacles);
	    }
	    //this could be moved to end
	    MPI_Reduce(&av_vel, &av_vels[ii], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        //if(rank==0) printf("Av vel: %.12E\n", av_vels[ii]);
            #ifdef DEBUG
            printf("==timestep: %d==\n", ii);
            printf("av velocity: %.12E\n", av_vels[ii]);
            printf("tot density: %.12E\n", total_density(params, cells_even));
            #endif
    }

    //Restore full grid
    if(ii%2 == 0) {
      MPI_Gatherv(cells_odd, expected_cells, MPI_SPEED_T, cells_whole, group_sizes,
		  displacements, MPI_SPEED_T, 0, MPI_COMM_WORLD); 
    }
    else {
      MPI_Gatherv(cells_even, expected_cells, MPI_SPEED_T, cells_whole, group_sizes,
		  displacements, MPI_SPEED_T, 0, MPI_COMM_WORLD); 
    }

    if(rank == 0) {
      params.ny = full_y;
      gettimeofday(&timstr,NULL);
      toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      getrusage(RUSAGE_SELF, &ru);
      timstr=ru.ru_utime;
      usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      timstr=ru.ru_stime;
      systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

      printf("Process %d ==done==\n", rank);
      printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params,av_vels[ii-1]));
      printf("Elapsed time:\t\t\t%.6f (s)\n", toc-tic);
      printf("Elapsed user CPU time:\t\t%.6f (s)\n", usrtim);
      printf("Elapsed system CPU time:\t%.6f (s)\n", systim);

      write_values(final_state_file, av_vels_file, params, cells_whole, obstacles_whole, av_vels);
      finalise(&cells_whole, &obstacles_whole, &av_vels);
    }
    finalise_worker(&cells_even, &cells_odd, &obstacles);
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

