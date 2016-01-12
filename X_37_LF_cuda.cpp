# include <stdio.h>
# include <mm_malloc.h>
# include <sys/time.h>
# include <math.h>

# define N_x37  6166
# define N_x37p	3230
# define cos40  0.76604444312
# define sin40  (-0.64278760969)
# define tranx	0.5
# define trany	(-0.293)
# define tranz	0.8

# define Ly		0.5
# define Lz		1.0
# define Lx		1.5
# define Ny		64
# define h		(Ly/(Ny-2))
# define Nz		(Ny*2)
# define Nx		(Ny*3)
# define Nxy	(Nx*Ny)
# define Nyz	(Ny*Nz)
# define Nt		(Nx*Ny*Nz)
# define Nt2	(Nt*2)
# define Nt3	(Nt*3)
# define Nt4	(Nt*4)
# define rat	0.00001	// dt/dx
# define tar	(1.0/rat)	// dx/dt
# define dt		(h*rat)

// http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
# define rhos	0.00008283 // (kg/m^3)
# define uxs	7000.0  // (m/s)
# define uys	0.0
# define uzs	0.0
# define ps		5.2	// (Pa)
// http://www.engineeringtoolbox.com/specific-heat-ratio-d_602.html
# define gamma	1.4
// https://en.wikipedia.org/wiki/Gas_constant#Specific_gas_constant
# define R		287.05
// anonymous state set in the body of shuttle
# define rhox	1.0
# define px	100000.0

# define sqr(x)	((x)*(x))

// threads per block for gpu kernal function
# define Tpb	256

int no_step = (Ny/100+1)*100000;

void inform();
void gpumalloc();
void gpufree();
void sendtogpu	(const float *h_air);
void sendbackgpu(float *h_U);
void sendbackgpu(const float *array, size_t sz);
void clockstart	(struct timeval *start, struct timeval *past);
void clockprint	(struct timeval start, int k, int total, struct timeval *past);
void clockend	(struct timeval start);
int  Load_X_37	(float *air);
void Air_X_37	(float *air,
				 const float *normal,
				 const float *edge1,
				 const float *edge2,
				 const float *edge3
				);
int  mesh_intersection_detection(float y,
							                   float z,
								                 float *xIntersec,
								           const float *normal,
								           const float *edge1,
								           const float *edge2,
								           const float *edge3
								           );
void ray_casting_algorithm(float *air, int index, const float *xIntersec, int mCount);
int check_nan_fail(float *d_U);
void save_data(const float *U);


__global__ void gpu_initialize	(float *U, const float *air);

__global__ void gpu_Fp_Gp_Hp(const float *U, const float *air, float *Fp, float *Gp, float *Hp);
__device__ void U_index(const float *U, float *Up, float *pUm, float air0, float airf, int shift);
__device__ void F_local(const float *U_loc, float *F_loc);
__device__ void G_local(const float *U_loc, float *G_loc);
__device__ void H_local(const float *U_loc, float *H_loc);
__device__ void flux_scheme(float *f_scheme, const float *Up, const float *pUm, const float *fluxp, const float *pfluxm);

__global__ void gpu_update_U(float *U, const float *air, const float *Fp, const float *Gp, const float *Hp);
__device__ float _delta_U(const float *Fp, const float *Gp, const float *Hp, int index, int eq);

__global__ void gpu_boundary(float *U);

float *air, *U, *Fp, *Gp, *Hp;

int main()
{
	size_t sz;
	float *h_air, *h_U;
	int block, step, block_bound, NaN_fail;
	struct timeval start, past;
	
	inform();
	// load mesh file and save to h_air
	sz = Nt*sizeof(float);
	h_air = (float *)_mm_malloc(sz, 32);
	if( Load_X_37(h_air)==-1 ) {
		_mm_free(h_air);
		printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		return -1;
	}
	// memory allocate
	gpumalloc();
	h_U = (float *)_mm_malloc(sz*5, 32);
	// send data to GPU
	sendtogpu(h_air);
	// initialize state on GPU
	printf("\nInitialize state on GPU...");
	block = (Nt+Tpb-1)/Tpb;
	gpu_initialize<<<block, Tpb>>>(U, air);
	printf("\tdone\n");
	// run time loops
	printf("\n\nRun time loops and calculate on GPU...\n");
    clockstart(&start, &past);
    block_bound = (Nx*Nz+Tpb-1)/Tpb;
	for(step=0; step<no_step; step++)
	{		
		gpu_Fp_Gp_Hp<<<block, Tpb>>>(U, air, Fp, Gp, Hp);
		gpu_update_U<<<block, Tpb>>>(U, air, Fp, Gp, Hp);
		gpu_boundary<<<block_bound, Tpb>>>(U);
		
		clockprint(start, step, no_step, &past);
		
		NaN_fail = check_nan_fail(U);
		if(NaN_fail) break;
	}
	if(NaN_fail) printf("\nNaN fail (TAT)...\n\n\n");
	else {
		clockend(start);
		printf("\nFinish~ Finish~ Finish~ Finish~\n\n");
	}
	
	sendbackgpu(h_U);
	save_data(h_U);
	
	gpufree();
	_mm_free(h_air);
	_mm_free(h_U);
	
	return 0;
}


void inform()
{
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	printf("The Flow Field over Suttle X-37 At Steady State\n\n");
	printf("Grid points:  Nx*Ny*Nz = %d*%d*%d = %.2e\n\n", Nx, Ny, Nz, (float)Nt);
	printf("Time steps:  %d (%.2f sec)\n\n", no_step, no_step*dt);
	printf("Courant Fourier Condition:  dt/dx = %f\n\n", rat);
	printf("\n\n");
}

void gpumalloc()
{
	size_t sz, sz5;
	printf("\nMemory allocating on GPU...");
	sz  = Nt*sizeof(float);
	sz5 = sz*5;
	int failflag = 0;
	if( cudaMalloc((void **)&U,  sz5) != cudaSuccess) failflag++;
	if( cudaMalloc((void **)&Fp, sz5) != cudaSuccess) failflag++;
	if( cudaMalloc((void **)&Gp, sz5) != cudaSuccess) failflag++;
	if( cudaMalloc((void **)&Hp, sz5) != cudaSuccess) failflag++;
	if( cudaMalloc((void **)&air, sz) != cudaSuccess) failflag++;
	if( cudaMalloc((void **)&phi, sz) != cudaSuccess) failflag++;
	if(failflag) printf("\n... Something failed...\n");
	else 		 printf("\tdone\n");
}


void gpufree()
{
	printf("\nMemory Freeing on GPU...");
	int failflag = 0;
	if( cudaFree(U)  != cudaSuccess) failflag++;
	if( cudaFree(Fp) != cudaSuccess) failflag++;
	if( cudaFree(Gp) != cudaSuccess) failflag++;
	if( cudaFree(Hp) != cudaSuccess) failflag++;
	if( cudaFree(air)!= cudaSuccess) failflag++;
	if( cudaFree(phi)!= cudaSuccess) failflag++;
	if(failflag) printf("\n... Something failed...\n");
	else 		 printf("\tdone\n");
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
}



void sendtogpu(const float *h_air)
{
	printf("\nSend data to GPU...");
	size_t sz = Nt*sizeof(float);
	if( cudaMemcpy(air, h_air, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("\n... Something failed...\n");
	} else	printf("\t\tdone\n");
}


void sendbackgpu(float *h_U)
{
	printf("\nSend data back from GPU...");
	size_t sz5 = 5*Nt*sizeof(float);
	if( cudaMemcpy(h_U, U, sz5, cudaMemcpyDeviceToHost) != cudaSuccess) {
			printf("\n... Something failed...\n");
	} else	printf("\tdone\n");
}

void clockstart(struct timeval *start, struct timeval *past)
{
	printf("............   0 %%  %8.3f sec  left %4d sec  predict %4d sec", 0.0, 0, 0);
	gettimeofday(start, NULL);
	gettimeofday(past,  NULL);
}

void clockprint(struct timeval start, int k, int total, struct timeval *past)
{
	struct timeval now;
	float dur, left;
	gettimeofday(&now, NULL);
	dur  =  (now.tv_sec-start.tv_sec)+(now.tv_usec-start.tv_usec)/1000000.0;
	left = ((now.tv_sec-past->tv_sec)+(now.tv_usec-past->tv_usec)/1000000.0)*(total-k);
	printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
    printf("%3d %%  %8.3f sec  left %4.0f sec  predict %4.0f sec", (k*100)/total, dur, left, dur+left);
    gettimeofday(past, NULL);
}

void clockend(struct timeval start)
{
	struct timeval now;
	float dur;
	gettimeofday(&now, NULL);
	dur  = (now.tv_sec-start.tv_sec)+(now.tv_usec-start.tv_usec)/1000000.0;
	printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
    printf("100 %%  %8.3f sec  left %4.0f sec  predict %4.0f sec\n\n", dur, 0.0, dur);
}

int Load_X_37(float *air)
{
    printf("Loading the mesh points of the model of X-37...\n");
    
    int i, k;
    float x, y, z;
    float *normal, *edge1, *edge2, *edge3;
    char v[20];
    FILE *fin;
    struct timeval start, past;
    size_t size_x37p;
    
    size_x37p = 3*(N_x37p+1)*sizeof(float);
    normal = (float *)_mm_malloc(size_x37p, 32);
    edge1  = (float *)_mm_malloc(size_x37p, 32);
    edge2  = (float *)_mm_malloc(size_x37p, 32);
    edge3  = (float *)_mm_malloc(size_x37p, 32);
    
    fin = fopen("X_37_Coarse_STL.txt", "r");
    if(!fin) {
    	printf("\nCouldn't find \"X_37_Coarse_STL.txt\" QwQ...\nplease check the file!\n\n\n");
    	return -1;
	}
    
    clockstart(&start, &past);
	// Load loop
    i = 0;
    for(k=0; k<N_x37; k++)
	{
        // load normal vector
		fscanf(fin, "%s%s%s%s%e%e%e", v, v, v, v, &x, &z, &y);
		// rotation
		normal[i]   = x*cos40-z*sin40;	// nx
		normal[i+1] = y;				// ny
		normal[i+2] = x*sin40+z*cos40;	// nz
		// load edge position
    	fscanf(fin, "%s%s%s%e%e%e", v, v, v, &x, &z, &y);
    	// rotation and translation
    	edge1[i]   = x*cos40-z*sin40+tranx;	// x1
    	edge1[i+1] = y+trany;				// y1
    	edge1[i+2] = x*sin40+z*cos40+tranz;	// z1
    	fscanf(fin, "%s%e%e%e", v, &x, &z, &y);
    	edge2[i]   = x*cos40-z*sin40+tranx; // x2
    	edge2[i+1] = y+trany;               // y2
    	edge2[i+2] = x*sin40+z*cos40+tranz; // z2
    	fscanf(fin, "%s%e%e%e", v, &x, &z, &y);
    	edge3[i]   = x*cos40-z*sin40+tranx; // x3
    	edge3[i+1] = y+trany;               // y3
    	edge3[i+2] = x*sin40+z*cos40+tranz; // z3
		// right half aircraft
		if(edge1[i+1]>0 || edge2[i+1]>0 || edge3[i+1]>0) i+=3;
	    // print progress
		clockprint(start, k, N_x37, &past);
    }
    clockend(start);
    fclose(fin);
    
    printf("\nTotal mesh number: %d\n", N_x37);
    printf("Right mesh number: %d\n", i/3);

    Air_X_37(air, normal, edge1, edge2, edge3);

    _mm_free(normal);
    _mm_free(edge1);
    _mm_free(edge2);   
    _mm_free(edge3);
    return 0;
}


void Air_X_37(float *air, const float *normal, const float *edge1, const float *edge2, const float *edge3)
{
	printf("\n\nGenerate the array of air-body detection...\n");
	struct timeval start, past;
    clockstart(&start, &past);
	
	int j, k, height, index, mCount;
	float y, z;
	float xIntersec[20] = {0.0}; // the x positon of the intersection of the mesh
	for(k=0; k<Nz; k++)
	{
		height = k*Nxy;
		z = (k-0.5)*h;
		for(j=0; j<Ny; j++)
		{
			index = height+j*Nx;
			y = (j-0.5)*h;
			mCount = mesh_intersection_detection(y, z, xIntersec, normal, edge1, edge2, edge3);
			ray_casting_algorithm(air, index, xIntersec, mCount);
		}
		clockprint(start, k, Nz, &past);
	}
	clockend(start);
}


int mesh_intersection_detection(float y, float z, float *xIntersec, const float *normal, const float *edge1, const float *edge2, const float *edge3)
{
	int Id, i = 0, mCount = 0, rCount;
	float ny, nz;	// normal vector torward right
	float ry, rz;	// relative position vector
	for(Id=0; Id<N_x37p; Id++)
	{
		// project the mesh point to yz plane,
		// and detect whether the point (y,z) is inside the projection of triangle mesh or not
		rCount = 0;	// accumulator for condition of right hand side point
		// edge1 -> edge2
		ny = edge2[i+2]-edge1[i+2]; //  vz
		nz = edge1[i+1]-edge2[i+1];	// -vy
		ry = y-edge1[i+1];
		rz = z-edge1[i+2];
		if(ny*ry+nz*rz>=0.0) rCount++;	// dot-product of normal vector and position vector
		// edge2 -> edge3
		ny = edge3[i+2]-edge2[i+2];
		nz = edge2[i+1]-edge3[i+1];
		ry = y-edge2[i+1];
		rz = z-edge2[i+2];
		if(ny*ry+nz*rz>=0.0) rCount++;
		// edge3 -> edge1
		ny = edge1[i+2]-edge3[i+2];
		nz = edge3[i+1]-edge1[i+1];
		ry = y-edge3[i+1];
		rz = z-edge3[i+2];
		if(ny*ry+nz*rz>=0.0) rCount++;
		// check the point (y,z) is inside or outside the projection of the mesh
		if(rCount==0 || rCount==3) {
			// solve the x cordinate of the intesection 
			// nx*(x-x1) + ny*(y-y1) + nz*(z-z1) = 0
			// x = x1 + ( ny*(y1-y)+nz*(z1-z) ) / nx
			xIntersec[mCount] = edge1[i] + ( normal[i+1]*(edge1[i+1]-y) + normal[i+2]*(edge1[i+2]-z) ) / normal[i];
			mCount++;
		}
		// update index
		i+=3;
	}
	// sorting the x coordinate of intersection points
	int loop;
	float regist;
	for(loop=mCount; loop>1; loop--) {
		for(i=1; i<loop; i++) {		
			if(xIntersec[i]<xIntersec[i-1]) {
				regist         = xIntersec[i  ];
				xIntersec[i  ] = xIntersec[i-1];
				xIntersec[i-1] = regist;
			}
		}
	}
	return mCount;
}


void ray_casting_algorithm(float *air, int index, const float *xIntersec, int mCount)
{
	int i = 0;
	int airflag = 1;
	int loop, bound;
	for(loop=0; loop<mCount; loop++) {
		bound = (int)( xIntersec[loop]/h+0.5 );
		while(i<=bound) {
			air[index+i] = (float)airflag;
			i++;
		}
		airflag = !airflag;
	}
	for(; i<Nx; i++) air[index+i] = 1.0;
}


int check_nan_fail(float *d_U)
{
	float check;
	int shift = 3.3*Nxy;
	cudaMemcpy(&check, d_U+shift, sizeof(float), cudaMemcpyDeviceToHost);
	if(check!=check)	return 1;
	else				return 0;
}




void save_data(const float *U)
{
	int i, j, k, height, row, index;
	float x, y, z, rho, ux, uy, uz, p, T;
	struct timeval start, past;
	FILE *fp;
	
	printf("\n\nSaving \"result.dat\" for Tecplot...\n");
	fp = fopen("result.dat","w");
    clockstart(&start, &past);
    fprintf(fp, "TITLE = \"The Flow Field over Suttle X37 At Steady State\"\r\n");
	fprintf(fp, "VARIABLES = \"X\", \"Y\", \"Z\", \"Density\", \"UX\", \"UY\", \"UZ\", \"Preasure\", \"Temperature\"\r\n");
    fprintf(fp, "ZONE I = %d, J = %d,  K = %d, F = POINT\r\n", Nx, Ny, Nz);
    for(k=0; k<Nz; k++) {
    	z = k*h;
		height = k*Nxy;
			for(j=0; j<Ny; j++) {
			y = j*h;
			row = height+j*Nx;
			for(i=0; i<Nx; i++) {
				x = i*h;
				index = row+i;
				rho = U[index    ];
				ux  = U[index+Nt ]/rho;
				uy  = U[index+Nt2]/rho;
				uz  = U[index+Nt3]/rho;
				p   = (gamma-1.0)*(U[index+Nt4]-0.5*rho*(sqr(ux)+sqr(uy)+sqr(uz)));	// Pa
				T   = p/(R*rho);	// K
				fprintf(fp, "%7.3f%7.3f%7.3f  %.3e%11.3f%11.3f%11.3f  %.3e%11.3f\r\n", x, y, z, rho, ux, uy, uz, p, T);
			}
		}
		clockprint(start, k, Nz, &past);
	}
	clockend(start);
	fclose(fp);
}



__global__ void gpu_initialize(float *U, const float *air)
{
	float air0, rho, p;
	int index = blockDim.x*blockIdx.x+threadIdx.x;
	if(index<Nt) {
		air0 = air[index];
		rho = (air0==1.0) ? rhos : rhox;
		U[index    ] = rho;
		U[index+Nt ] = air0*rhos*uxs;
		U[index+Nt2] = air0*rhos*uys;
		U[index+Nt3] = air0*rhos*uzs;
		p = (air0==1.0) ? ps : px;
		U[index+Nt4] = ( p/(gamma-1) + 0.5*rho*(air0*sqr(uxs)+sqr(uys)+sqr(uzs)) );
	}	
}

__global__ void gpu_Fp_Gp_Hp(const float *U, const float *air, float *Fp, float *Gp, float *Hp)
{
	int index, i, j, k;
	//int index0, forward;
	//float air0, airf;
	float Up[5], pUm[5], fluxp[5], pfluxm[5];
	index = blockDim.x*blockIdx.x+threadIdx.x;
	k = index/Nxy;
	j = (index-k*Nxy)/Nx;
	i = index-k*Nxy-j*Nx;
	if((k<Nz-1) && (j<Ny-1) && (i<Nx-1)) {
		// x-direction
		U_index(&U[index], Up, pUm, air[index], air[index+1], 1);
		F_local( Up,  fluxp);
		F_local(pUm, pfluxm);
		flux_scheme(&Fp[index], Up, pUm, fluxp, pfluxm);
		// y-direction
		U_index(&U[index], Up, pUm, air[index], air[index+Nx], Nx);
		G_local( Up,  fluxp);
		G_local(pUm, pfluxm);
		flux_scheme(&Gp[index], Up, pUm, fluxp, pfluxm);
		// z-direction
		U_index(&U[index], Up, pUm, air[index], air[index+Nxy], Nxy);
		H_local( Up,  fluxp);
		H_local(pUm, pfluxm);
		flux_scheme(&Hp[index], Up, pUm, fluxp, pfluxm);
	}
}

__device__ void U_index(const float *U, float *Up, float *pUm, float air0, float airf, int shift)
{
	int index0, forward;
	index0  = (air0==1.0) ? 0 : shift;
	forward = (airf==0.0) ? 0 : shift;
	air0 = 2.0*air0-1.0;
	airf = 2.0*airf-1.0;
	Up[0]  =      U[index0    ];
	Up[1]  = air0*U[index0+Nt ];
	Up[2]  = air0*U[index0+Nt2];
	Up[3]  = air0*U[index0+Nt3];
	Up[4]  =      U[index0+Nt4];
	pUm[0] =      U[forward    ];
	pUm[1] = airf*U[forward+Nt ];
	pUm[2] = airf*U[forward+Nt2];
	pUm[3] = airf*U[forward+Nt3];
	pUm[4] =      U[forward+Nt4];
}

__device__ void F_local(const float *U_loc, float *F_loc)
{
	float rho, ux, uy, uz, E, p;
	rho = U_loc[0];
	ux  = U_loc[1]/rho;
	uy  = U_loc[2]/rho;
	uz  = U_loc[3]/rho;
	E   = U_loc[4];
	p   = (gamma-1.0)*(E-0.5*rho*(sqr(ux)+sqr(uy)+sqr(uz)));
	F_loc[0] = rho*ux;
	F_loc[1] = rho*sqr(ux)+p;
	F_loc[2] = rho*ux*uy;
	F_loc[3] = rho*ux*uz;
	F_loc[4] = (E+p)*ux;
}

__device__ void G_local(const float *U_loc, float *G_loc)
{
	float rho, ux, uy, uz, E, p;
	rho = U_loc[0];
	ux  = U_loc[1]/rho;
	uy  = U_loc[2]/rho;
	uz  = U_loc[3]/rho;
	E   = U_loc[4];
	p   = (gamma-1.0)*(E-0.5*rho*(sqr(ux)+sqr(uy)+sqr(uz)));
	G_loc[0] = rho*uy;
	G_loc[1] = rho*ux*uy;
	G_loc[2] = rho*sqr(uy)+p;
	G_loc[3] = rho*uy*uz;
	G_loc[4] = (E+p)*uy;
}

__device__ void H_local(const float *U_loc, float *H_loc)
{
	float rho, ux, uy, uz, E, p;
	rho = U_loc[0];
	ux  = U_loc[1]/rho;
	uy  = U_loc[2]/rho;
	uz  = U_loc[3]/rho;
	E   = U_loc[4];
	p   = (gamma-1.0)*(E-0.5*rho*(sqr(ux)+sqr(uy)+sqr(uz)));
	H_loc[0] = rho*uz;
	H_loc[1] = rho*ux*uz;
	H_loc[2] = rho*uy*uz;
	H_loc[3] = rho*sqr(uz)+p;
	H_loc[4] = (E+p)*uz;
}

__device__ void flux_scheme(float *f_scheme, const float *Up, const float *pUm, const float *fluxp, const float *pfluxm)
{
    // Lax-Friedrichs flux
	f_scheme[0  ] = 0.5*( fluxp[0]+pfluxm[0] + tar*(Up[0]-pUm[0])/3.0 );
	f_scheme[Nt ] = 0.5*( fluxp[1]+pfluxm[1] + tar*(Up[1]-pUm[1])/3.0 );
	f_scheme[Nt2] = 0.5*( fluxp[2]+pfluxm[2] + tar*(Up[2]-pUm[2])/3.0 );
	f_scheme[Nt3] = 0.5*( fluxp[3]+pfluxm[3] + tar*(Up[3]-pUm[3])/3.0 );
	f_scheme[Nt4] = 0.5*( fluxp[4]+pfluxm[4] + tar*(Up[4]-pUm[4])/3.0 );
}


__global__ void gpu_update_U(float *U, const float *air, const float *Fp, const float *Gp, const float *Hp)
{
	int index, i, j, k;
	bool valid;
	index = blockDim.x*blockIdx.x+threadIdx.x;
	k = index/Nxy;
	j = (index-k*Nxy)/Nx;
	i = index-k*Nxy-j*Nx;
	valid = (k<Nz-1)*(j<Ny-1)*(i<Nx-1)*(i>0)*(j>0)*(k>0)*(air[index]==1.0);
	if(valid) {
		U[index    ] += _delta_U(Fp, Gp, Hp, index, 0);
		U[index+Nt ] += _delta_U(Fp, Gp, Hp, index, 1);
		U[index+Nt2] += _delta_U(Fp, Gp, Hp, index, 2);
		U[index+Nt3] += _delta_U(Fp, Gp, Hp, index, 3);
		U[index+Nt4] += _delta_U(Fp, Gp, Hp, index, 4);
	}
}

__device__ float _delta_U(const float *Fp, const float *Gp, const float *Hp, int index, int eq)
{
	index += eq*Nt;
	return rat*( Fp[index-1]-Fp[index] + Gp[index-Nx]-Gp[index] + Hp[index-Nxy]-Hp[index] );
}


__global__ void gpu_boundary(float *U)
{
	int index, i, j, k;
	index = blockDim.x*blockIdx.x+threadIdx.x;
	k = index/Nx;
	i = index-k*Nx;
	if(k<Nz) {
		// plane y=0 (reflective boundary)
		index = k*Nxy+i;
		U[index    ] =  U[index    +Nx];
		U[index+Nt ] =  U[index+Nt +Nx];
		U[index+Nt2] = -U[index+Nt2+Nx];
		U[index+Nt3] =  U[index+Nt3+Nx];
		U[index+Nt4] =  U[index+Nt4+Nx];
		// plane y=Ly (free boundary)
		index = k*Nxy+(Ny-1)*Nx+i;
		U[index    ] =  U[index    -Nx];
		U[index+Nt ] =  U[index+Nt -Nx];
		U[index+Nt2] =  U[index+Nt2-Nx];
		U[index+Nt3] =  U[index+Nt3-Nx];
		U[index+Nt4] =  U[index+Nt4-Nx];
		
		if(k<Ny) {
			j = k;
			// plane z=0 (free boundary)
			index = j*Nx+i;
			U[index    ] =  U[index    +Nxy];
			U[index+Nt ] =  U[index+Nt +Nxy];
			U[index+Nt2] =  U[index+Nt2+Nxy];
			U[index+Nt3] =  U[index+Nt3+Nxy];
			U[index+Nt4] =  U[index+Nt4+Nxy];
			// plane z=Lz (free boundary)
			index = (Nz-1)*Nxy+j*Nx+i;
			U[index    ] =  U[index    -Nxy];
			U[index+Nt ] =  U[index+Nt -Nxy];
			U[index+Nt2] =  U[index+Nt2-Nxy];
			U[index+Nt3] =  U[index+Nt3-Nxy];
			U[index+Nt4] =  U[index+Nt4-Nxy];
			
			if(i<Nz) {
				k = i;
				// plane x=Lx (free boundary)
				index = k*Nxy+j*Nx+(Nx-1);
				U[index    ] =  U[index    -1];
				U[index+Nt ] =  U[index+Nt -1];
				U[index+Nt2] =  U[index+Nt2-1];
				U[index+Nt3] =  U[index+Nt3-1];
				U[index+Nt4] =  U[index+Nt4-1];				
			}
		}
	}
		
				
}
