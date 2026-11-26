#include <psum/psum.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace psum::prelude;
using namespace psum::particle_collision;
constexpr int N=100000,B=24,L=2048,J=256;
constexpr double m=9.109e-31,qel=1.602176634e-19,n=1e19,dt=2e-10;
struct V {double x,p;}; // axial speed and transverse speed squared
struct Both {double real=0,mw=0;};
uint64_t hash64(uint64_t x){x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
struct R {uint64_t key; double operator()(){key+=0x9e3779b97f4a7c15ULL;return ((hash64(key)>>12)+.5)*0x1p-52;}};
double square(V a){return a.x*a.x+a.p;}
double meanF(double g,double ld){
 if(g<=1e-10)return 0;
 constexpr double mass1=m,mass2=m,charge1=-qel,charge2=-qel,EPSILON0=8.854187817e-12,H_BAR=1.054571817e-34,density=n*(2.-1./N);const double sigma_max=coulomb_cross_section_limit(n),un=g,lambdaD=ld;
        const double m12 = mass1 * mass2 / (mass1 + mass2);
        const double b_min = std::max(H_BAR / (m12 * un), std::abs(charge1 * charge2) / (un * un * 4 * M_PI * EPSILON0 * m12));
        const double lambda = std::max(lambdaD / b_min, 1.01);
        const double s_uncapped = std::log(lambda) / 4 / M_PI *
            std::pow((charge1 * charge2) / (EPSILON0 * m12), 2) *
            density * std::pow(un, -3) * dt;
        const double s = std::min(s_uncapped, sigma_max * density * un * dt);


 if(s<=.01)return -s*std::expm1(-2./s);
 double A=solve_A(s);if(A<=.01)return 1-A/3;
 return 1-1/std::tanh(A)+1/A;
}
double heat(V a,V b,double cosphi,double ld){
 double ap=std::sqrt(a.p),bp=std::sqrt(b.p),dx=a.x-b.x;
 double g=std::sqrt(dx*dx+(ap-bp)*(ap-bp)+2*ap*bp*(1-cosphi));
 return (square(a)-square(b))*meanF(g,ld);
}
int main(int argc,char** argv){ const uint64_t salt=hash64(std::stoull(argv[1]));
 sycl::queue queue;std::cerr<<queue.get_device().get_info<sycl::info::device::name>()<<'\n';
 std::ofstream out("heat.csv");out<<std::setprecision(16)<<"step,batch,hot_eV,cold_eV,real_gap_loss_eV_us,maxwell_gap_loss_eV_us,kurt_hot,kurt_cold\n";
 const double factor=m/(3*qel*dt)*1e-6*N/(2.*N-1);
 psum::field::device_array<V> maxw(queue,2*B*N), states(queue,2*N);
 psum::field::device_array<double> norms(queue,2*B*L),scales(queue,2*B);
 psum::field::device_array<Both> sums(queue,B*L);
 auto M=maxw.data();auto U=states.data();auto Z=norms.data();auto S=scales.data();auto F=sums.data();
 for(int step=0;step<=200000;step+=(step<80000?2000:10000)){
  double temp[2],kurt[2];std::vector<V> data(2*N);
  for(int side=0;side<2;side++){
   std::ifstream in("snapshots/"+std::to_string(step)+"-"+std::to_string(side)+".bin",std::ios::binary);
   if(!in.read((char*)(data.data()+side*N),N*sizeof(V)))throw std::runtime_error("missing snapshot");
   double a=0,b=0;for(int j=0;j<N;j++){double v2=square(data[side*N+j]);a+=v2;b+=v2*v2;}
   temp[side]=a/N*m/(3*qel);kurt[side]=.6*b*N/(a*a)-1;
  }
  queue.memcpy(U,data.data(),data.size()*sizeof(V)).wait();
  double ld=std::sqrt(8.854187817e-12/(qel*n*(1/temp[0]+1/temp[1])));
  queue.parallel_for(sycl::range<1>(2*B*L),[=](sycl::id<1> idx){
   int z=idx[0],group=z/L,lane=z%L;R r{hash64(((uint64_t(step)<<32)+z)^salt)};
   double sum=0; for(int i=lane;i<N;i+=L){
    double x=std::sqrt(-2*std::log(r()))*std::cos(2*M_PI*r());double p=-2*std::log(r());
    M[group*N+i]={x,p};sum+=x*x+p;
   } Z[z]=sum;
  }).wait();
  auto norm=norms.to_host();std::vector<double> scale(2*B);
  for(int g=0;g<2*B;g++){double total=0;for(int j=0;j<L;j++)total+=norm[g*L+j];
    scale[g]=std::sqrt(temp[g%2]*3*qel*N/(m*total));}
  queue.memcpy(S,scale.data(),scale.size()*sizeof(double)).wait();
  queue.parallel_for(sycl::range<1>(B*L),[=](sycl::id<1> idx){
   int z=idx[0],batch=z/L;R r{hash64(0xa04f1732ef920700ULL^(uint64_t(step)<<32)^z^salt)};
   Both v;double sh=S[2*batch],sc=S[2*batch+1];
   for(int k=0;k<J;k++){int i=int(N*r()),j=int(N*r());double phi=std::cos(2*M_PI*r());
    v.real+=heat(U[i],U[N+j],phi,ld);
    V hot=M[2*batch*N+i],cold=M[(2*batch+1)*N+j];
    v.mw+=heat({hot.x*sh,hot.p*sh*sh},{cold.x*sc,cold.p*sc*sc},phi,ld);
   }
   F[z]=v;
  }).wait();
  auto ans=sums.to_host();for(int b=0;b<B;b++){Both v;
   for(int j=0;j<L;j++){v.real+=ans[b*L+j].real;v.mw+=ans[b*L+j].mw;}
   out<<step<<','<<b<<','<<temp[0]<<','<<temp[1]<<','<<v.real*factor/(J*L)<<','<<v.mw*factor/(J*L)<<','<<kurt[0]<<','<<kurt[1]<<'\n';
  }out.flush();
 }
}
