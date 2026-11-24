#include "search.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>

using namespace dc;
int main(int argc,char**argv){
 if(argc!=7)return 2;
 try{
  auto a=std::make_unique<NNUE>(argv[1]),b=std::make_unique<NNUE>(argv[2]);
  Search engines[2]={Search(*a),Search(*b)};
  uint64_t seed=std::stoull(argv[3]);int pairs=std::stoi(argv[4]);double ms=std::stod(argv[5]);
  if(pairs<1||pairs>10000||!std::isfinite(ms)||ms<=0)throw std::runtime_error("bad limits");
  std::ofstream replays(argv[6]);if(!replays)throw std::runtime_error("cannot write replay");
  std::mt19937_64 rng(seed);
  std::vector<double> scores;
  std::array<uint64_t,2> nodes{},moves{},depth_zero{};
  for(int pair=0;pair<pairs;++pair){
   std::vector<int> bag;
   for(int p=1;p<=14;++p)for(int n=0;n<inventory[p-1];++n)bag.push_back(p);
   std::shuffle(bag.begin(),bag.end(),rng);
   int opening=int(std::uniform_int_distribution<int>(0,31)(rng)),color_first=color(bag[opening]);
   int start=int(rng()%2);double score=0;
   for(int g=0;g<2;++g){
    const int first=(start+g)%2;
    Board board;std::vector<int> history{opening*32+opening};
    if(!board.play(history[0],bag[opening]))throw std::runtime_error("opening");
    while(!board.over()&&board.ply<4096){
     int player=board.turn==color_first?first:1-first;
     auto result=engines[player].analyze(board,{12,100000000,ms});
     int m=result.move,from=m/32,to=m%32;
     if(!board.legal(m)||!board.play(m,from==to?bag[from]:0))throw std::runtime_error("illegal decision");
     nodes[player]+=result.nodes;++moves[player];depth_zero[player]+=result.depth==0;history.push_back(m);
    }
    if(!board.over())throw std::runtime_error("incomplete pair");
    score+=board.draw?0.5:((board.winner==color_first?first:1-first)==1?1.:0.);
    replays<<"{\"id\":\"ablation-"<<seed<<'-'<<pair<<'-'<<first<<"\",\"ver\":\"v2.1.0-perpetual-chase\",\"hid\":[";
    for(size_t i=0;i<bag.size();++i){if(i)replays<<',';replays<<bag[i]-1;}
    replays<<"],\"mov\":[";
    for(size_t i=0;i<history.size();++i){if(i)replays<<',';replays<<history[i];}
    replays<<"],\"res\":"<<(board.draw?0:board.winner==RED?1:-1)<<",\"ply\":"<<board.ply<<"}\n";
    replays.flush();if(!replays)throw std::runtime_error("write failure");
   }
   scores.push_back(score);std::cerr<<"pair "<<pair+1<<'/'<<pairs<<": "<<score<<'\n';
  }
  std::cout<<std::setprecision(12)<<"{\"seed\":"<<seed<<",\"planned_pairs\":"<<pairs<<",\"budget_ms\":"<<ms<<",\"pair_scores\":[";
  for(size_t i=0;i<scores.size();++i){if(i)std::cout<<',';std::cout<<scores[i];}
  std::cout<<"],\"incomplete_games\":0,\"nodes\":["<<nodes[0]<<','<<nodes[1]<<"],\"moves\":["<<moves[0]<<','<<moves[1]<<"],\"depth_zero\":["<<depth_zero[0]<<','<<depth_zero[1]<<"]}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
