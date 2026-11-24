#pragma once
#include "nnue.hpp"
#ifdef DC_EXPERIMENTAL_DELTA
#include "nnue_delta.hpp"
#endif
#ifdef DC_EXPERIMENTAL_TT
#include "exact_cache.hpp"
#endif
#include <chrono>
#include <limits>
#include <map>
#include <numeric>

namespace dc {
struct Limits {int depth=12; uint64_t nodes=100000000; double ms=250;};
struct Analysis {int move=-1,depth=0;uint64_t nodes=0;double ms=0,value=0;std::map<int,double> roots;uint64_t cache_hits=0;};
struct BudgetExceeded{};
class Search {
 using Clock=std::chrono::steady_clock;
 const NNUE& model;
 Limits limits;
 Clock::time_point started;
 uint64_t nodes=0,cache_hits=0;
#ifdef DC_EXPERIMENTAL_TT
 ExactCache cache;
#endif
 static constexpr double INF=std::numeric_limits<double>::infinity();
 void check(){
  if(nodes>=limits.nodes||std::chrono::duration<double,std::milli>(Clock::now()-started).count()>=limits.ms)throw BudgetExceeded{};
 }
 void tick(){check();++nodes;}
 static double terminal(const Board& b){return b.draw?0:b.winner==RED?1:-1;}
 static std::vector<int> ordered(const Board& b,const std::map<int,double>* prior=nullptr){
  auto moves=b.moves();
  auto rank=[&](int m){return m/32==m%32?1:b.grid[m%32]!=EMPTY?2:0;};
  std::sort(moves.begin(),moves.end(),[&](int a,int c){
   if(prior&&!prior->empty()){
    double va=prior->at(a),vc=prior->at(c);
    if(va!=vc)return b.turn==BLACK?va<vc:va>vc;
   }
   int x=rank(a),y=rank(c);return x!=y?x>y:a<c;
  });return moves;
 }
 double child_value(const Board& b,const Features& f,const Accumulator& acc,int m,int reveal,int depth,double alpha,double beta){
  check();
  Board child=b;if(!child.play(m,reveal))throw std::logic_error("illegal search child");
#ifdef DC_EXPERIMENTAL_DELTA
  (void)f;
  return node(child,Features{},advance_move(model,acc,b,child,m),depth-1,alpha,beta);
#else
  auto next=features(child);
  return node(child,next,model.advance(acc,f,next),depth-1,alpha,beta);
#endif
 }
 double branch(const Board& b,const Features& f,const Accumulator& a,int m,int depth,double alpha,double beta){
  if(m/32!=m%32)return child_value(b,f,a,m,0,depth,alpha,beta);
  const int total=std::accumulate(b.remaining.begin(),b.remaining.end(),0);
  if(total<=0)throw std::logic_error("empty belief");
  // Full probability enumeration. Bound children only via mathematically valid
  // chance windows, never arbitrary width or identity truncation.
  double sum=0,seen=0;
  for(int i=0;i<14;++i)if(b.remaining[i]){
   double p=double(b.remaining[i])/total,remaining=std::max(0.,1-seen-p);
   // Opening is valued from the first mover's eventual color, not RED.
   bool opening=b.turn==NONE;
   double ca=opening||!std::isfinite(alpha)?-INF:(alpha-sum-remaining)/p;
   double cb=opening||!std::isfinite(beta)?INF:(beta-sum+remaining)/p;
   if(ca>1+1e-12)return alpha;
   if(cb<-1-1e-12)return beta;
   double v=child_value(b,f,a,m,i+1,depth,ca<=-1?-INF:ca,cb>=1?INF:cb);
   if(opening&&color(i+1)==BLACK)v=-v;
   if(v<=ca&&std::isfinite(ca))return alpha;
   if(v>=cb&&std::isfinite(cb))return beta;
   sum+=p*v;seen+=p;
   double left=std::max(0.,1-seen);
   if(sum+left<=alpha)return sum+left;
   if(sum-left>=beta)return sum-left;
  }
  return std::clamp(sum,-1.,1.);
 }
 double node(const Board& b,const Features& f,const Accumulator& a,int depth,double alpha,double beta){
  tick();
  if(b.over())return terminal(b);
  if(depth<=0)return model.value(a);
#ifdef DC_EXPERIMENTAL_TT
  const bool eligible=cacheable(b,depth);
  ExactKey key{};
  const double initial_alpha=alpha,initial_beta=beta;
  if(eligible){
   key=exact_key(b,depth);
   if(auto found=cache.find(key);found!=cache.end()){++cache_hits;return found->second;}
  }
#endif
  auto moves=ordered(b);
  if(moves.empty())throw std::logic_error("nonterminal without legal moves");
  bool maximize=b.turn!=BLACK;
  double v=maximize?-INF:INF;
  for(int m:moves){
   double score=branch(b,f,a,m,depth,alpha,beta);
   if(maximize){v=std::max(v,score);alpha=std::max(alpha,v);}
   else {v=std::min(v,score);beta=std::min(beta,v);}
   if(alpha>=beta)break;
  }
#ifdef DC_EXPERIMENTAL_TT
  if(eligible&&v>initial_alpha&&v<initial_beta&&cache.size()<32768)cache.emplace(key,v);
#endif
  return v;
 }
public:
 explicit Search(const NNUE& m):model(m){}
 Analysis analyze(const Board& b,Limits lim={}){
  if(lim.depth<1||lim.nodes<1||!(lim.ms>0))throw std::invalid_argument("invalid search limits");
  limits=lim;nodes=0;cache_hits=0;started=Clock::now();Analysis result;
#ifdef DC_EXPERIMENTAL_TT
  cache.clear();
#endif
  auto root=ordered(b);if(root.empty()){if(b.over())result.value=terminal(b);return result;}
  result.move=root.front();auto f=features(b);auto acc=model.init(f);result.value=model.value(acc);
  for(int d=1;d<=lim.depth;++d){
   std::map<int,double> values;double best=b.turn==BLACK?INF:-INF;int move=root.front();
   try{
    for(int m:root){
     double v=branch(b,f,acc,m,d,-INF,INF);values[m]=v;
     if((b.turn==BLACK?v<best:v>best)||(v==best&&m<move)){best=v;move=m;}
    }
   }catch(const BudgetExceeded&){break;}
   result.depth=d;result.move=move;result.value=best;result.roots=std::move(values);
   root=ordered(b,&result.roots);
  }
  result.nodes=nodes;result.cache_hits=cache_hits;result.ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();return result;
 }
};
}
