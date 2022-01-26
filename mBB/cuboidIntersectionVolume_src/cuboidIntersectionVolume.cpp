/* 
mex gpc.c cuboidIntersectionVolume.c -O -output cuboidIntersectionVolume              % optimized
mex gpc.c cuboidIntersectionVolume.c -argcheck -output cuboidIntersectionVolume       % with argument checking
mex gpc.c cuboidIntersectionVolume.c -g -output cuboidIntersectionVolume              % for debugging 
*/

// #include "mex.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gpc.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include <typeinfo>

void gpc_polygon_clip(gpc_op           set_operation,
                              gpc_polygon     *subject_polygon,
                              gpc_polygon     *clip_polygon,
                              gpc_polygon     *result_polygon);


namespace py = pybind11;

/* ===============================
	    Constants
	===============================*/

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

bool  isColMajor(py::array_t<double> m){
    return m.strides(0) < m.strides(1);
}
bool  isRowMajor(py::array_t<double> m){
    return m.strides(0) > m.strides(1);
}

/*	=================================
		GATEWAY ROUTINE TO MATLAB
	=================================*/

py::array_t<double, py::array::c_style> cuboidIntersectionVolume(
    std::array<py::array_t<double>,2> _prhs)
{
    //any major to column major
    std::array<py::array_t<double, py::array::f_style>,2> prhs;// = _prhs;
    for(int i=0;i<_prhs.size();++i) prhs[i]=_prhs[i];

    py::array_t<double, py::array::f_style> plhs;
    unsigned int i,j,n1,n2,c,v,m;
    double* volume;
    double* b1;
    double* b2;
    unsigned int joffset;
    double zOverlap;
    double areaOverlap;
    double* result_vertex;
    gpc_polygon subject, clip, result;
    gpc_vertex_list subject_contour;
    gpc_vertex_list clip_contour;
    int hole = 0;
    
    subject.num_contours = 1;
    subject.hole = &hole;
    subject.contour = &subject_contour;
    subject.contour[0].num_vertices = 4;
    
    clip.num_contours = 1;
    clip.hole = &hole;
    clip.contour = &clip_contour;
    clip.contour[0].num_vertices = 4;
    
    n2 = prhs[0].shape(1); //mxGetN(prhs[0]);
    n1 = prhs[1].shape(1); //mxGetN(prhs[1]);

    //colmajor
    //x x x
    //y y y
    b2 = prhs[0].mutable_data(); //mxGetPr(prhs[0]);
    b1 = prhs[1].mutable_data(); //mxGetPr(prhs[1]);

    //plhs[0] = mxCreateNumericMatrix(n2, n1, mxDOUBLE_CLASS, mxREAL);
    plhs.resize({n2,n1});
    volume = plhs.mutable_data();
    
    for (i=0; i<n1; ++i){
        /* subject */
        subject.contour[0].vertex = (gpc_vertex *)b1;
        for (j=0; j<n2; ++j){
            joffset = 10*j;
            zOverlap = MIN(b1[9],b2[joffset+9]) - MAX(b1[8],b2[joffset+8]);

            if (zOverlap>0){
                /* get intersection */ 
                clip.contour[0].vertex = (gpc_vertex *)(b2+joffset);

                gpc_polygon_clip(gpc_op::GPC_INT, &subject, &clip, &result);
                
                if (result.num_contours>0 && result.contour[0].num_vertices > 2) { 
                    /* compute area of intersection */

                    /*
                     * http://www.mathopenref.com/coordpolygonarea.html
                     * Green's theorem for the functions -y and x; 
                     http://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon
                     */
                    result_vertex = (double*)(result.contour[0].vertex);
                    m = result.contour[0].num_vertices;
                    areaOverlap = (result_vertex[2*m-2]*result_vertex[1]-result_vertex[2*m-1]*result_vertex[0]);
                    for (v= 1; v < m; v++)
                    {
                        areaOverlap += (result_vertex[v*2-2]*result_vertex[v*2+1]-result_vertex[v*2-1]*result_vertex[v*2]);
                    }
                    *volume = zOverlap * 0.5 * fabs(areaOverlap);
                    
                    
                }
                gpc_free_polygon(&result);
            }            
            ++volume;
        }
        b1+=10;
    }
    /*
    gpc_free_polygon(&subject);
    gpc_free_polygon(&clip);
    */

    //column major to row major
    return py::array_t<double, py::array::c_style>(plhs);
}

PYBIND11_MODULE(cuboidIntersectionVolume, m) {
    m.doc() = "cuboidIntersectionVolume";

#if !defined(NDEBUG)
    m.attr("debug_enabled") = true;
#else
    m.attr("debug_enabled") = false;
#endif

    m.def(
        "cuboidIntersectionVolume",
        cuboidIntersectionVolume,
        "cuboidIntersectionVolume",
        py::arg("prhs"));

}

