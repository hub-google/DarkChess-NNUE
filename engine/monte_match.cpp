#include "monte_carlo.hpp"
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
using namespace dc;
struct Totals{uint64_t moves=0,nodes=0,simulations=0,depth_sum=0,over_budget=0;double ms=0;};
int main(int argc,char**argv){
 if(argc!=9)return 2;
 try{
  auto model=std::make_unique<NNUE>(argv[1]);
  auto policy=std::make_unique<Policy>(argv[2]);
  const std::array<std::string,2> modes{argv[3],argv[4]};
  for(auto& s:modes)if(s!="expecti"&&s!="mcts"&&s!="puct")throw std::runtime_error("unknown mode");
  uint64_t seed=std::stoull(argv[5]);int pairs=std::stoi(argv[6]);double ms=std::stod(argv[7]);
  if(pairs<1||pairs>1000||!std::isfinite(ms)||ms<=0)throw std::runtime_error("bad limits");
  std::ofstream out(argv[8]);if(!out)throw std::runtime_error("replay output");
  std::mt19937_64 deals(seed);std::vector<double> scores;std::array<Totals,2> totals{};
  Search reference(*model);
  for(int i=0;i<pairs;++i){
   std::vector<int> bag;for(int p=1;p<=14;++p)for(int n=0;n<inventory[p-1];++n)bag.push_back(p);
   std::shuffle(bag.begin(),bag.end(),deals);
   const int opening=std::uniform_int_distribution<int>(0,31)(deals),first_color=color(bag[opening]),order=int(deals()%2);
   double score=0;
   for(int g=0;g<2;++g){
    int first=(order+g)%2;
    Board b;std::vector<int> history{opening*32+opening};
    if(!b.play(history.front(),bag[opening]))throw std::runtime_error("opening");
    // Search random streams are independent of the sampled hidden permutation.
    std::array<std::unique_ptr<MonteCarlo>,2> mcts;
    for(int p=0;p<2;++p)if(modes[p]!="expecti")mcts[p]=std::make_unique<MonteCarlo>(*model,modes[p]=="puct"?policy.get():nullptr,seed+1000003ULL*i+17*g+p);
    while(!b.over()&&b.ply<4096){
     int p=b.turn==first_color?first:1-first,m=-1;auto& t=totals[p];
     if(modes[p]=="expecti"){
      auto a=reference.analyze(b,{12,100000000,ms});m=a.move;t.nodes+=a.nodes;t.depth_sum+=a.depth;t.ms+=a.ms;t.over_budget+=a.ms>2*ms;
     }else{
      auto a=mcts[p]->analyze(b,ms);m=a.move;t.nodes+=a.nodes;t.simulations+=a.simulations;t.depth_sum+=a.max_tree_depth;t.ms+=a.ms;t.over_budget+=a.ms>2*ms;
     }
     ++t.moves;
     if(!b.legal(m)||!b.play(m,m/32==m%32?bag[m/32]:0))throw std::runtime_error("illegal choice");
     history.push_back(m);
    }
    if(!b.over())throw std::runtime_error("incomplete match");
    score+=b.draw?.5:((b.winner==first_color?first:1-first)==1?1.:0.);
    out<<"{\"id\":\"monte-"<<seed<<'-'<<i<<'-'<<g<<"\",\"ver\":\"v2.1.0-perpetual-chase\",\"hid\":[";
    for(size_t j=0;j<bag.size();++j){if(j)out<<',';out<<bag[j]-1;}
    out<<"],\"mov\":[";
    for(size_t j=0;j<history.size();++j){if(j)out<<',';out<<history[j];}
    out<<"],\"res\":"<<(b.draw?0:b.winner==RED?1:-1)<<",\"ply\":"<<b.ply<<"}\n";
    out.flush();if(!out)throw std::runtime_error("write failure");
   }
   scores.push_back(score);std::cerr<<"pair "<<i+1<<'/'<<pairs<<": "<<score<<'\n';
  }
  std::cout<<std::setprecision(12)<<"{\"seed\":"<<seed<<",\"planned_pairs\":"<<pairs<<",\"budget_ms\":"<<ms<<",\"baseline\":\""<<modes[0]<<"\",\"candidate\":\""<<modes[1]<<"\",\"pair_scores\":[";
  for(size_t i=0;i<scores.size();++i){if(i)std::cout<<',';std::cout<<scores[i];}
  std::cout<<"],\"incomplete_games\":0,\"stats\":[";
  for(int p=0;p<2;++p){
   if(p)std::cout<<',';
   auto t=totals[p];
   std::cout<<"{\"moves\":"<<t.moves<<",\"nodes\":"<<t.nodes<<",\"simulations\":"<<t.simulations<<",\"ms\":"<<t.ms<<",\"over_budget_2x\":"<<t.over_budget<<",\"depth_sum\":"<<t.depth_sum<<"}";
  }
  std::cout<<"]}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
