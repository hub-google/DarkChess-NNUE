#include "search.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>

using namespace dc;
struct Label {
 int move,depth; uint64_t nodes; double ms,q,v;
 std::vector<std::pair<int,double>> pi;
};
template<class T> void array(std::ostream& out,const std::vector<T>& a){
 out<<'[';bool first=true;for(const auto& v:a){if(!first)out<<',';first=false;out<<v;}out<<']';
}
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"model path, model version, seed, games, ms, output required\n";return 2;}
 try{
  auto model=std::make_unique<NNUE>(argv[1]);Search engine(*model);
  const std::string version=argv[2];
  if(version.size()!=17||version.substr(0,5)!="nnue-"||version.find_first_not_of("0123456789abcdef",5)!=std::string::npos)throw std::runtime_error("bad model identifier");
  uint64_t seed=std::stoull(argv[3]);int count=std::stoi(argv[4]);double ms=std::stod(argv[5]);
  if(count<1||count>10000||!(ms>0)||!std::isfinite(ms))throw std::runtime_error("bad experiment arguments");
  std::ofstream out(argv[6]);if(!out)throw std::runtime_error("cannot write output");
  out<<std::setprecision(12);
  std::mt19937_64 rng(seed);
  int total=0,depth0=0;uint64_t nodes=0;double search_ms=0;
  auto start=std::chrono::steady_clock::now();
  for(int game=0;game<count;++game){
   // This bag is confined to the referee; Search takes Board, which has no hidden layout.
   std::vector<int> bag;
   for(int p=1;p<=14;++p)for(int n=0;n<inventory[p-1];++n)bag.push_back(p);
   std::shuffle(bag.begin(),bag.end(),rng);
   Board board;std::vector<Label> labels;
   while(!board.over()&&board.ply<4096){
    int side=board.turn;
    auto a=engine.analyze(board,{12,100000000,ms});
    if(!board.legal(a.move))throw std::runtime_error("illegal engine move");
    Label label{a.move,a.depth,a.nodes,a.ms,side==NONE?0.:a.value,model->value(model->init(features(board))),{}};
    if(a.depth>0&&side!=NONE){
     // Normalize only the top 16 legal root values. This is an experimental
     // policy teacher, not an assertion that visits are MCTS probabilities.
     auto rank=[&](double v){return side==RED?v:-v;};
     std::vector<std::pair<int,double>> ranked(a.roots.begin(),a.roots.end());
     std::sort(ranked.begin(),ranked.end(),[&](auto x,auto y){return x.second==y.second?x.first<y.first:rank(x.second)>rank(y.second);});
     if(ranked.size()>16)ranked.resize(16);
     double best=rank(ranked.front().second),sum=0;
     for(auto [m,v]:ranked){double w=std::exp((rank(v)-best)/0.20);label.pi.emplace_back(m,w);sum+=w;}
     for(auto& row:label.pi)row.second/=sum;
     // Diverse early games; only near-best moves are sampled, not arbitrary
     // random actions. q remains the search optimum, not the sampled action.
     if(board.ply<=12){
      std::vector<double> weights;
      for(auto [m,v]:ranked){(void)m;weights.push_back(rank(v)>=best-0.04?std::exp((rank(v)-best)/0.20):0.);}
      std::discrete_distribution<size_t> pick(weights.begin(),weights.end());label.move=ranked[pick(rng)].first;
     }
    }else if(side==NONE&&a.depth>0){
     std::vector<int> best;for(auto [m,v]:a.roots)if(v>=a.value-1e-7)best.push_back(m);
     if(!best.empty())label.move=best[std::uniform_int_distribution<size_t>(0,best.size()-1)(rng)];
    }
    int from=label.move/32,to=label.move%32;
    if(!board.play(label.move,from==to?bag[from]:0))throw std::runtime_error("referee rejected move");
    ++total;depth0+=a.depth==0;nodes+=a.nodes;search_ms+=a.ms;labels.push_back(std::move(label));
   }
   if(!board.over())throw std::runtime_error("incomplete game, no manufactured draw");
   out<<"{\"id\":\"v2-native-"<<seed<<'-'<<game<<"\",\"ver\":\"v2.1.0-perpetual-chase\",\"ts\":"
      <<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
      <<",\"seed\":"<<seed<<",\"model\":\""<<version<<"\",\"search_mode\":\"native-full-chance-timed\",\"budget_ms\":"<<ms<<",\"hid\":";
   for(int& p:bag)--p;
   array(out,bag);
   auto field=[&](const char* name,auto get){out<<",\""<<name<<"\":[";bool first=true;for(const auto& a:labels){if(!first)out<<',';first=false;out<<get(a);}out<<']';};
   field("mov",[](const Label& x){return x.move;});field("qd",[](const Label& x){return x.depth;});
   field("q",[](const Label& x){return x.q;});field("v",[](const Label& x){return x.v;});
   field("qn",[](const Label& x){return x.nodes;});field("qt",[](const Label& x){return x.ms;});
   out<<",\"pi\":[";
   for(size_t i=0;i<labels.size();++i){if(i)out<<',';out<<'[';bool first=true;for(auto [m,p]:labels[i].pi){if(!first)out<<',';first=false;out<<'['<<m<<','<<p<<']';}out<<']';}
   out<<"],\"res\":"<<(board.draw?0:board.winner==RED?1:-1)<<",\"ply\":"<<board.ply<<"}\n";
   out.flush();if(!out)throw std::runtime_error("failed to save complete game");
   std::cerr<<"complete "<<game+1<<'/'<<count<<" plies "<<board.ply<<'\n';
  }
  const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cout<<std::setprecision(12)<<"{\"games\":"<<count<<",\"seed\":"<<seed<<",\"budget_ms\":"<<ms<<",\"moves\":"<<total<<",\"depth_zero\":"<<depth0<<",\"nodes\":"<<nodes<<",\"search_ms\":"<<search_ms<<",\"wall_seconds\":"<<wall<<"}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
