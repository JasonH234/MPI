#ifndef LBM_HDR_FILE
#define LBM_HDR_FILE

#define NSPEEDS         9

/* Size of box in imaginary 'units */
#define BOX_X_SIZE (100.0)
#define BOX_Y_SIZE (100.0)

/* struct to hold the parameter values */
typedef struct {
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
    int nx;            /* no. of cells in x-direction */
    int ny;            /* no. of cells in y-direction */
    int max_iters;      /* no. of iterations */
    int tot_cells;
    int reynolds_dim;  /* dimension for Reynolds number */
    int minX;
    int maxX;
    int minY;
    int maxY;
} param_t;

/* obstacle positions */
typedef struct {
    float obs_x_min;
    float obs_x_max;
    float obs_y_min;
    float obs_y_max;
} obstacle_t;

/* struct to hold the 'speed' values */
typedef struct {
    float speeds[NSPEEDS];
} speed_t;

typedef enum { ACCEL_ROW, ACCEL_COLUMN } accel_e;
typedef struct {
    accel_e col_or_row;
    int idx;
} accel_area_t;

/* Parse command line arguments to get filenames */
void parse_args (int argc, char* argv[],
    char** final_state_file, char** av_vels_file, char** param_file);

void initialise(const char* paramfile, accel_area_t * accel_area,
    param_t* params, speed_t** cells_ptr,
    unsigned int** obstacles_ptr, float** av_vels_ptr);

void initialise_worker(param_t params, speed_t** cells_even_ptr, 
	speed_t** cells_odd_ptr, unsigned int** obstacles_ptr, const int expected_cells);

void initialise_unused(param_t params, speed_t** cells_ptr);

void write_values(const char * final_state_file, const char * av_vels_file,
    const param_t params, speed_t* cells, const unsigned int* obstacles, float* av_vels);

void finalise(speed_t** cells_whole_ptr, unsigned int** obstacles_whole_ptr, float** av_vels_ptr);
void finalise_worker(speed_t** cells_ptr, speed_t** tmp_cells_ptr, unsigned int** obstacles_ptr);

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    speed_t* cells, const unsigned int* obstacles);

void copy_cells(const param_t params, speed_t* cells1, speed_t* cells2);
float simulation_steps(const param_t *params, speed_t* cells, const speed_t* old_cells, const unsigned int* obstacles);
/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const param_t params, speed_t* cells);


/* calculate Reynolds number */
float calc_reynolds(const param_t params, const float av_vel);

/* Exit, printing out formatted string */
#define DIE(...) exit_with_error(__LINE__, __FILE__, __VA_ARGS__)
void exit_with_error(int line, const char* filename, const char* format, ...)
__attribute__ ((format (printf, 3, 4)));

#endif
