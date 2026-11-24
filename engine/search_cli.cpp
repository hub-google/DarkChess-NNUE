#include "search.hpp"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
using namespace dc;
// One persistent model. Input contains only revealed move history, never the hidden bag.
int main(int argc,char**argv){
 if(argc!=2)return 2;
 try{
  auto model=std::make_unique<NNUE>(argv[1]);
  Search search(*model);
  std::cout<<"READY"<<std::endl;
  std::string line;
  while(std::getline(std::cin,line)){
   std::istringstream in(line);
   int n,depth;uint64_t nodes;double ms;
   if(!(in>>n>>depth>>nodes>>ms)||n<0||n>4096||depth<1||nodes<1||!std::isfinite(ms)||ms<=0)throw std::runtime_error("bad request");
   const auto total_start=std::chrono::steady_clock::now();
   const std::clock_t cpu_start=std::clock();
   Board b;
   for(int i=0;i<n;++i){int m,r;if(!(in>>m>>r)||!b.play(m,r))throw std::runtime_error("bad public history");}
   std::string extra;if(in>>extra)throw std::runtime_error("trailing request data");
   if(b.over())throw std::runtime_error("terminal search request");
   auto a=search.analyze(b,{depth,nodes,ms});
   if(!b.legal(a.move))throw std::runtime_error("illegal search result");
   double cpu_ms=1000.*double(std::clock()-cpu_start)/CLOCKS_PER_SEC;
   double total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-total_start).count();
   std::cout<<std::setprecision(12)<<"{\"move\":"<<a.move<<",\"depth\":"<<a.depth<<",\"nodes\":"<<a.nodes<<",\"ms\":"<<a.ms<<",\"cpu_ms\":"<<cpu_ms<<",\"total_ms\":"<<total_ms<<",\"value\":"<<a.value<<"}"<<std::endl;
  }
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
