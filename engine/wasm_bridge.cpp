#include "search.hpp"
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

namespace {
std::unique_ptr<dc::NNUE> model;
std::string response;
}
extern "C" {
int dc_init(){
 try{model=std::make_unique<dc::NNUE>("/champion.bin");return 1;}
 catch(...){model.reset();return 0;}
}
const char* dc_analyze(const char* request){
 try{
  if(!model||!request)throw std::runtime_error("not initialized");
  std::istringstream in(request);
  int n=0,depth=0;uint64_t nodes=0;double ms=0;
  if(!(in>>n>>depth>>nodes>>ms)||n<0||n>4096||depth<1||depth>12||!nodes||!std::isfinite(ms)||ms<=0)
   throw std::runtime_error("invalid request");
  dc::Board board;
  for(int i=0;i<n;++i){int m,r;if(!(in>>m>>r)||!board.play(m,r))throw std::runtime_error("invalid public history");}
  std::string extra;if(in>>extra||board.over())throw std::runtime_error("invalid search state");
  dc::Search engine(*model);
  auto a=engine.analyze(board,{depth,nodes,ms});
  if(!board.legal(a.move))throw std::runtime_error("illegal move");
  std::ostringstream out;
  out<<std::setprecision(15)<<"{\"move\":"<<a.move<<",\"depth\":"<<a.depth<<",\"nodes\":"<<a.nodes<<",\"ms\":"<<a.ms<<",\"value\":"<<a.value<<",\"roots\":{";
  bool first=true;for(auto [m,v]:a.roots){if(!first)out<<',';first=false;out<<'"'<<m<<"\":"<<v;}
  out<<"}}";response=out.str();
 }catch(...){response="{\"error\":\"invalid WASM search request or engine failure\"}";}
 return response.c_str();
}
}
