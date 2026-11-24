#pragma once
#include "search.hpp"
#include <memory>
#include <random>
namespace dc {
// Public-state UCT and optional policy-prior PUCT. Chance is sampled from the
// remaining inventory, never from the referee's private permutation.
class Policy {
 std::array<std::array<float,498>,64> w1{};
 std::array<float,64> b1{};
 std::array<std::array<float,64>,1024> w2{};
 std::array<float,1024> b2{};
public:
 explicit Policy(const char* path){
  std::ifstream in(path,std::ios::binary);
  if(!in)throw std::runtime_error("policy model open");
  auto read=[&](){
   std::array<unsigned char,4> b{};
   if(!in.read(reinterpret_cast<char*>(b.data()),4))throw std::runtime_error("policy truncated");
   uint32_t bits=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
   float x=std::bit_cast<float>(bits);if(!std::isfinite(x))throw std::runtime_error("policy nonfinite");return x;
  };
  for(auto& r:w1)for(auto& x:r)x=read();
  for(auto& x:b1)x=read();
  for(auto& r:w2)for(auto& x:r)x=read();
  for(auto& x:b2)x=read();
  if(in.peek()!=EOF)throw std::runtime_error("policy extra bytes");
 }
 std::vector<double> priors(const Board& b,const std::vector<int>& moves)const{
  auto f=features(b);std::array<double,64> h{};
  for(int i=0;i<64;++i){
   double z=b1[i];for(int j=0;j<498;++j)if(f[j]!=0)z+=double(w1[i][j])*f[j];
   h[i]=std::max(0.,z);
  }
  std::vector<double> out;out.reserve(moves.size());
  for(int m:moves){
   double z=b2[m];for(int i=0;i<64;++i)z+=double(w2[m][i])*h[i];out.push_back(z);
  }
  double maximum=*std::max_element(out.begin(),out.end()),total=0;
  for(auto& v:out){v=std::exp(v-maximum);total+=v;}
  for(auto& v:out)v/=total;
  return out;
 }
};
struct MonteResult{int move=-1;uint64_t simulations=0,nodes=0;int max_tree_depth=0;double ms=0;};
class MonteCarlo {
 struct Node;
 struct Edge {
  int move;double prior;uint64_t visits=0;double sum=0;
  std::array<std::unique_ptr<Node>,15> outcomes{};
  Edge(int m,double p):move(m),prior(p){}
 };
 struct Node {uint64_t visits=0;std::vector<Edge> edges;};
 using Clock=std::chrono::steady_clock;
 const NNUE& model;
 const Policy* policy;
 std::mt19937_64 rng;
 Clock::time_point start;
 double budget=0;
 uint64_t nodes=0,limit=0;
 int max_depth=0;
 void check(){
  if(nodes>=limit||std::chrono::duration<double,std::milli>(Clock::now()-start).count()>=budget)throw BudgetExceeded{};
 }
 int reveal(const Board& b){
  int total=std::accumulate(b.remaining.begin(),b.remaining.end(),0);
  if(total<=0)throw std::logic_error("empty belief");
  int n=std::uniform_int_distribution<int>(0,total-1)(rng);
  for(int p=0;p<14;++p){n-=b.remaining[p];if(n<0)return p+1;}
  throw std::logic_error("invalid belief");
 }
 void expand(Node& n,const Board& b){
  auto moves=b.moves();if(moves.empty())throw std::logic_error("nonterminal has no legal moves");
  std::sort(moves.begin(),moves.end());
  auto priors=policy?policy->priors(b,moves):std::vector<double>(moves.size(),1./moves.size());
  n.edges.reserve(moves.size());
  for(size_t i=0;i<moves.size();++i)n.edges.emplace_back(moves[i],priors[i]);
 }
 double simulate(const Board& b,const Features& f,const Accumulator& a,Node& n,int depth){
  check();++nodes;max_depth=std::max(max_depth,depth);
  if(b.over()){++n.visits;return b.draw?0.:b.winner==RED?1.:-1.;}
  if(depth>=32){++n.visits;return model.value(a);}
  if(n.edges.empty()){expand(n,b);++n.visits;return model.value(a);}
  const double sign=b.turn==RED?1.:-1.;
  Edge* selected=nullptr;double best=-std::numeric_limits<double>::infinity();
  for(auto& e:n.edges){
   double q=e.visits?sign*e.sum/double(e.visits):0.;
   double score=policy?q+1.5*e.prior*std::sqrt(double(n.visits+1))/(1+e.visits):
     e.visits?q+1.41421356237*std::sqrt(std::log(double(n.visits+1))/e.visits):std::numeric_limits<double>::infinity();
   if(!selected||score>best||(score==best&&e.move<selected->move)){selected=&e;best=score;}
  }
  auto& edge=*selected;
  int p=edge.move/32==edge.move%32?reveal(b):0;
  Board child=b;if(!child.play(edge.move,p))throw std::logic_error("illegal simulation");
  auto& outcome=edge.outcomes[p];
  if(!outcome)outcome=std::make_unique<Node>();
  auto next=features(child);
  double v=simulate(child,next,model.advance(a,f,next),*outcome,depth+1);
  ++edge.visits;edge.sum+=v;++n.visits;return v;
 }
public:
 MonteCarlo(const NNUE& m,const Policy* p,uint64_t seed):model(m),policy(p),rng(seed){}
 MonteResult analyze(const Board& b,double ms,uint64_t node_limit=100000000){
  if(b.turn==NONE||b.over()||!(ms>0)||!node_limit)throw std::invalid_argument("known-side nonterminal required");
  start=Clock::now();budget=ms;nodes=0;limit=node_limit;max_depth=0;
  Node root;expand(root,b);
  auto f=features(b);auto acc=model.init(f);
  MonteResult result;result.move=root.edges.front().move;
  for(;;){
   try{(void)simulate(b,f,acc,root,0);++result.simulations;}
   catch(const BudgetExceeded&){break;}
  }
  const double sign=b.turn==RED?1.:-1.;
  uint64_t visits=0;double mean=-std::numeric_limits<double>::infinity();
  for(const auto& e:root.edges)if(e.visits){
   double q=sign*e.sum/e.visits;
   if(e.visits>visits||(e.visits==visits&&(q>mean||(q==mean&&e.move<result.move)))){
    result.move=e.move;visits=e.visits;mean=q;
   }
  }
  result.nodes=nodes;result.max_tree_depth=max_depth;result.ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  return result;
 }
};
}
