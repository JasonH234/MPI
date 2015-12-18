/* Functions pertinent to the outer simulation steps */

#include <math.h>

#include "lbm.h"

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    speed_t* cells, const unsigned int* obstacles)
{
    int ii,jj;     /* generic counters */

    /* compute weighting factors */
    const float w1 = params.density * params.accel / 9.0f;
    const float w2 = params.density * params.accel / 36.0f;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;

        for (ii = 0; ii < params.ny; ii++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[4] - w1) > 0.0f &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0f &&
            (cells[ii*params.nx + jj].speeds[8] - w2) > 0.0f )
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

        for (jj = params.minX; jj < params.maxX; jj++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[3] - w1) > 0.0f &&
            (cells[ii*params.nx + jj].speeds[6] - w2) > 0.0f &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0f )
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

float simulation_steps(const param_t *params, speed_t* cells, const speed_t* old_cells, const unsigned int* obstacles)
{
  int n;            /* generic counters */
    const float c_sq = 1.0f/3.0f;  /* square of speed of sound */
    const float w0 = 4.0f/9.0f;    /* weighting factor */
    const float w1 = 1.0f/9.0f;    /* weighting factor */
    const float w2 = 1.0f/36.0f;   /* weighting factor */
    const float omega_dif = 1.0f-params->omega;
    
    float d_equ[NSPEEDS];        /* equilibrium densities */
    float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */
    
    n = params->nx*params->ny;
    while (n--)
    {
        if(obstacles[n]==2)
            continue;
        float tmp[NSPEEDS];
        const unsigned int y_s = (n < params->nx) ? n + params->nx*(params->ny+1) : n - params->nx;
        const unsigned int y_n = n + params->nx;
        const unsigned int x_e = (((n+1)%params->nx)==0) ? 1-params->nx : 1;            
        const unsigned int x_w = ((n%params->nx)==0) ? params->nx-1 : -1;
       // if(n == 0)printf("%d %d %d %d se %d sw %d ne %d nw %d \n", y_s, y_n,x_e,x_w, (y_s+x_e), (y_s+x_w), (y_n+x_e), (y_n+x_w));
        tmp[0]  = old_cells[n].speeds[0]; /* central cell, */                                                     /* no movement   */
        tmp[1] = old_cells[n+x_w].speeds[1]; /* east */
        tmp[2]  = old_cells[y_s].speeds[2]; /* north */
        tmp[3] = old_cells[n+x_e].speeds[3]; /* west */
        tmp[4]  = old_cells[y_n].speeds[4]; /* south */
        tmp[5] = old_cells[y_s + x_w].speeds[5]; 
        tmp[6] = old_cells[y_s + x_e].speeds[6]; 
        tmp[7] = old_cells[y_n + x_e].speeds[7]; 
        tmp[8] = old_cells[y_n + x_w].speeds[8];
            
        if (obstacles[n])
          {
	            cells[n].speeds[1] = tmp[3];
                cells[n].speeds[2] = tmp[4];
                cells[n].speeds[3] = tmp[1];
                cells[n].speeds[4] = tmp[2];
                cells[n].speeds[5] = tmp[7];
                cells[n].speeds[6] = tmp[8];
                cells[n].speeds[7] = tmp[5];
                cells[n].speeds[8] = tmp[6];
          } 
        else {
	    const float local_density = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + tmp[8];
        const float local_inverse = 1.0f/local_density;
        const float u_x = (tmp[1] +
                tmp[5] +
                tmp[8]
            - (tmp[3] +
                tmp[6] +
                tmp[7]))
            * local_inverse;

        const float u_y = (tmp[2] +
                tmp[5] +
                tmp[6]
            - (tmp[4] +
                tmp[7] +
                tmp[8]))
            * local_inverse;

        const float u_sq = u_x * u_x + u_y * u_y;
	    const float c1 = params->omega*local_density *w1;
	    const float c2 = params->omega* local_density * w2;
	    const float u_sum = u_x + u_y;
	    const float u_dif = u_x - u_y;
	    const float d = 1.0f - u_sq*1.5f;

        /*d_equ[0] = params->omega*w0 * local_density * d;
        d_equ[1] = c1 * (d + 4.5*u_x*(2.0/3.0 + u_x));
        d_equ[3] = c1 * (d - 4.5*u_x*(2.0/3.0 - u_x));
        d_equ[2] = c1 * (d + 4.5*u_y*(2.0/3.0 + u_y)); 
        d_equ[4] = c1 * (d - 4.5*u_y*(2.0/3.0 - u_y));
        d_equ[5] = c2 * (d + 4.5*u_sum*(2.0/3.0 + u_sum));
        d_equ[7] = c2 * (d - 4.5*u_sum*(2.0/3.0 - u_sum));
        d_equ[8] = c2 * (d + 4.5*u_dif*(2.0/3.0 + u_dif));
        d_equ[6] = c2 * (d - 4.5*u_dif*(2.0/3.0 - u_dif));*/
             cells[n].speeds[0] = (tmp[0]*omega_dif +params->omega*w0 * local_density * d);
             cells[n].speeds[1] = (tmp[1]*omega_dif +c1 * (d + 4.5f*u_x*(2.0f/3.0f + u_x)));
             cells[n].speeds[2] = (tmp[2]*omega_dif +c1 * (d + 4.5f*u_y*(2.0f/3.0f + u_y)));
             cells[n].speeds[3] = (tmp[3]*omega_dif +c1 * (d - 4.5f*u_x*(2.0f/3.0f - u_x)));
             cells[n].speeds[4] = (tmp[4]*omega_dif +c1 * (d - 4.5f*u_y*(2.0f/3.0f - u_y)));
             cells[n].speeds[5] = (tmp[5]*omega_dif +c2 * (d + 4.5f*u_sum*(2.0f/3.0f + u_sum)));
             cells[n].speeds[6] = (tmp[6]*omega_dif +c2 * (d - 4.5f*u_dif*(2.0f/3.0f - u_dif)));
             cells[n].speeds[7] = (tmp[7]*omega_dif +c2 * (d - 4.5f*u_sum*(2.0f/3.0f - u_sum)));
             cells[n].speeds[8] = (tmp[8]*omega_dif +c2 * (d + 4.5f*u_dif*(2.0f/3.0f + u_dif)));
/*
        for (kk = 0; kk < NSPEEDS; kk++)
        {
             //d_equ[kk] = 0.0f;
             cells[ii*params->nx+jj].speeds[kk] = (tmp[kk]*omega_dif +d_equ[kk]);
        }*/
        tot_u += sqrt(u_sq);
      }
    }
    return tot_u / (float) params->tot_cells; 
}

