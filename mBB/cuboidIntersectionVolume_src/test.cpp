#include<pybind11/embed.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<stdio.h>

namespace py=pybind11;

int main(int argc, char **argv){
    py::scoped_interpreter guard{};
    py::module_ sys = py::module_::import("sys");
    py::print(sys.attr("path"));
    py::module_ numpy = py::module_::import("numpy");

    py::array_t<double> da(pybind11::array::ShapeContainer{2,3});
    da.mutable_at(0,0)=0;
    da.mutable_at(0,1)=1;
    da.mutable_at(0,2)=2;
    da.mutable_at(1,0)=3;
    da.mutable_at(1,1)=4;
    da.mutable_at(1,2)=5;
    printf("default : %d %d\n", da.strides(0), da.strides(1));
    double *pda=(double*)da.data();
    for(int i=0;i<da.size();++i) printf("%lf ", pda[i]);
    printf("\n");

    py::array_t<double, py::array::f_style> fa({2,3});
    fa.mutable_at(0,0)=0;
    fa.mutable_at(0,1)=1;
    fa.mutable_at(0,2)=2;
    fa.mutable_at(1,0)=3;
    fa.mutable_at(1,1)=4;
    fa.mutable_at(1,2)=5;
    printf("f_style : %d %d\n", fa.strides(0), fa.strides(1));
    double *pfa=(double*)fa.data();
    for(int i=0;i<fa.size();++i) printf("%lf ", pfa[i]);
    printf("\n");

    py::array_t<double, py::array::c_style> ca({2,3});
    ca.mutable_at(0,0)=0;
    ca.mutable_at(0,1)=1;
    ca.mutable_at(0,2)=2;
    ca.mutable_at(1,0)=3;
    ca.mutable_at(1,1)=4;
    ca.mutable_at(1,2)=5;
    printf("c_style : %d %d\n", ca.strides(0), ca.strides(1));
    double *pca=(double*)ca.data();
    for(int i=0;i<ca.size();++i) printf("%lf ", pca[i]);
    printf("\n");

    fa = ca;
    py::print(fa);

    return 0;
}