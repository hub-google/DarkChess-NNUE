#include "search.hpp"
#include <iomanip>
#include <iostream>
#include <memory>
using namespace dc;
int main(int argc,char** argv){
 if(argc!=2)return 2;
 auto model=std::make_unique<NNUE>(argv[1]);Search engine(*model);
 int n,roots=0;double error=0;uint64_t nodes=0;double ms=0;
 if(!(std::cin>>n)||n<=0)return 2;
 for(int i=0;i<n;++i){
  Board b;int path;std::cin>>path;
  for(int j=0;j<path;++j){int m,r;std::cin>>m>>r;if(!b.play(m,r))return 1;}
  int depth,k;std::cin>>depth>>k;std::map<int,double> expected;
  for(int j=0;j<k;++j){int m;double v;std::cin>>m>>v;expected[m]=v;}
  if(!std::cin)return 2;
  auto a=engine.analyze(b,{depth,100000000,60000});
  if(a.depth!=depth||a.roots.size()!=expected.size()||!b.legal(a.move)){
   std::cerr<<"search completion mismatch "<<i<<" depth "<<a.depth<<" expected "<<depth<<"\n";return 1;
  }
  for(auto [m,v]:expected){
   if(!a.roots.contains(m))return 1;
   double e=std::abs(a.roots.at(m)-v);error=std::max(error,e);
   if(e>1e-5){std::cerr<<"search score mismatch "<<i<<" move "<<m<<" error "<<e<<"\n";return 1;}++roots;
  }
  auto limited=engine.analyze(b,{12,1,1000});
  if(!b.legal(limited.move)||limited.nodes>1||limited.depth!=0||!limited.roots.empty())return 1;
  auto timed=engine.analyze(b,{12,100000000,0.001});
  if(!b.legal(timed.move)||(timed.depth==0&&!timed.roots.empty()))return 1;
  nodes+=a.nodes;ms+=a.ms;
 }
 std::cout<<std::setprecision(12)<<"{\"positions\":"<<n<<",\"root_scores\":"<<roots<<",\"max_error\":"<<error<<",\"nodes\":"<<nodes<<",\"ms\":"<<ms<<",\"budget_fallback\":true,\"parity\":true}\n";
}
