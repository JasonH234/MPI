/* Functions pertinent to the outer simulation steps */

#include <math.h>

#include "lbm.h"

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    speed_t* cells, int* obstacles)
{
    int ii,jj;     /* generic counters */
    double w1,w2;  /* weighting factors */

    /* compute weighting factors */
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;

        for (ii = 0; ii < params.ny; ii++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[4] - w1) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                cells[ii*params.nx + jj].speeds[2] += w1;
                cells[ii*params.nx + jj].speeds[5] += w2;
                cells[ii*params.nx + jj].speeds[6] += w2;
                /* decrease 'south-side' densities */
                cells[ii*params.nx + jj].speeds[4] -= w1;
                cells[ii*params.nx + jj].speeds[7] -= w2;
                cells[ii*params.nx + jj].speeds[8] -= w2;
            }
        }
    }
    else
    {
        ii = accel_area.idx;

        for (jj = 0; jj < params.nx; jj++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[3] - w1) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[6] - w2) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                cells[ii*params.nx + jj].speeds[1] += w1;
                cells[ii*params.nx + jj].speeds[5] += w2;
                cells[ii*params.nx + jj].speeds[8] += w2;
                /* decrease 'west-side' densities */
                cells[ii*params.nx + jj].speeds[3] -= w1;
                cells[ii*params.nx + jj].speeds[6] -= w2;
                cells[ii*params.nx + jj].speeds[7] -= w2;
            }
        }
    }
}

float simulation_steps(const param_t params, speed_t* cells, const speed_t* old_cells, int* obstacles)
{
  int ii,jj, kk;            /* generic counters */
    const float c_sq = 1.0/3.0;  /* square of speed of sound */
    const float w0 = 4.0/9.0;    /* weighting factor */
    const float w1 = 1.0/9.0;    /* weighting factor */
    const float w2 = 1.0/36.0;   /* weighting factor */

    float u_x,u_y;               /* av. velocities in x and y directions */
    float u_sq;                  /* squared velocity */
    float local_density;         /* sum of densities in a particular cell */
    float u[NSPEEDS];            /* directional velocities */
    float d_equ[NSPEEDS];        /* equilibrium densities */
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u = 0;          /* accumulated magnitudes of velocity for each cell */

    /* loop over _all_ cells */
#pragma omp parallel for reduction(+:tot_cells,tot_u) shared(cells, old_cells, obstacles) private(ii, jj, kk, u_x, u_y, u_sq, local_density, u, d_equ) default(none) schedule(auto)
    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
	    float tmp[NSPEEDS];
            int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */
            y_n = (ii + 1) % params.ny;
            x_e = (jj + 1) % params.nx;
            y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

            tmp[0]  = old_cells[ii*params.nx + jj].speeds[0]; /* central cell, */                                                     /* no movement   */
            tmp[1] = old_cells[ii*params.nx + x_w].speeds[1]; /* east */
            tmp[2]  = old_cells[y_s*params.nx + jj].speeds[2]; /* north */
            tmp[3] = old_cells[ii*params.nx + x_e].speeds[3]; /* west */
            tmp[4]  = old_cells[y_n*params.nx + jj].speeds[4]; /* south */
            tmp[5] = old_cells[y_s*params.nx + x_w].speeds[5]; /* north-east */
            tmp[6] = old_cells[y_s*params.nx + x_e].speeds[6]; /* north-west */
            tmp[7] = old_cells[y_n*params.nx + x_e].speeds[7]; /* south-west */
            tmp[8] = old_cells[y_n*params.nx + x_w].speeds[8]; /* south-east */
            
	    if (obstacles[ii*params.nx + jj])
	      {
		cells[ii*params.nx + jj].speeds[1] = tmp[3];
                cells[ii*params.nx + jj].speeds[2] = tmp[4];
                cells[ii*params.nx + jj].speeds[3] = tmp[1];
                cells[ii*params.nx + jj].speeds[4] = tmp[2];
                cells[ii*params.nx + jj].speeds[5] = tmp[7];
                cells[ii*params.nx + jj].speeds[6] = tmp[8];
                cells[ii*params.nx + jj].speeds[7] = tmp[5];
                cells[ii*params.nx + jj].speeds[8] = tmp[6];
	      } 
	    else
	      {
		local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp[kk];
                }

                u_x = (tmp[1] +
                        tmp[5] +
                        tmp[8]
                    - (tmp[3] +
                        tmp[6] +
                        tmp[7]))
                    / local_density;

                u_y = (tmp[2] +
                        tmp[5] +
                        tmp[6]
                    - (tmp[4] +
                        tmp[7] +
                        tmp[8]))
                    / local_density;

                u_sq = u_x * u_x + u_y * u_y;

                u[1] =   u_x; 
                u[2] =         u_y;
                u[3] = - u_x;      
                u[4] =       - u_y;
                u[5] =   u_x + u_y;
                u[6] = - u_x + u_y;
                u[7] = - u_x - u_y;
                u[8] =   u_x - u_y;

                d_equ[0] = w0 * local_density * (1.0 - u_sq / (2.0 * c_sq));
                d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
                    + (u[1] * u[1]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq
                    + (u[2] * u[2]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[3] = w1 * local_density * (1.0 + u[3] / c_sq
                    + (u[3] * u[3]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[4] = w1 * local_density * (1.0 + u[4] / c_sq
                    + (u[4] * u[4]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
                    + (u[5] * u[5]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq
                    + (u[6] * u[6]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[7] = w2 * local_density * (1.0 + u[7] / c_sq
                    + (u[7] * u[7]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[8] = w2 * local_density * (1.0 + u[8] / c_sq
                    + (u[8] * u[8]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));

                for (kk = 0; kk < NSPEEDS; kk++)
                {
		  cells[ii*params.nx+jj].speeds[kk] = (tmp[kk] + params.omega * 
                        (d_equ[kk] - tmp[kk]));
                }
                tot_u += sqrt(u_x*u_x + u_y*u_y);
                ++tot_cells;
	      }
        }
    }
    return tot_u / (float)tot_cells;
}

