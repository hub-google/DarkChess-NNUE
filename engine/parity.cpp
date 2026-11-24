#include "nnue.hpp"
#include <iomanip>
#include <iostream>
#include <memory>
using namespace dc;
int main(int argc,char** argv){
 if(argc!=2)return 2;
 auto model=std::make_unique<NNUE>(argv[1]);
 int games,positions=0,terminal=0,chase=0,repeated=0;double max_error=0,max_increment_error=0;
 if(!(std::cin>>games)||games<=0)return 2;
 for(int game=0;game<games;++game){
  Board b;int count;
  if(!(std::cin>>count>>b.turn>>b.quiet))return 2;
  for(int& v:b.grid)std::cin>>v;
  for(int& v:b.remaining)std::cin>>v;
  Features f=features(b);Accumulator a=model->init(f);
  for(int ply=0;ply<count;++ply){
   int turn,winner,draw,quiet,rep,k;double expected;
   std::cin>>turn>>winner>>draw>>quiet>>rep>>expected;
   Grid g{},tokens{};std::array<int,14> remaining{};
   for(int& x:g)std::cin>>x;
   for(int& x:tokens)std::cin>>x;
   for(int& x:remaining)std::cin>>x;
   int captured;std::cin>>captured;std::vector<int> cap(captured);for(int& x:cap)std::cin>>x;
   auto read_chase=[](){Chase c;int size;std::cin>>c.attacker>>c.target>>c.count>>c.chaser>>size;c.route.resize(size);for(int& v:c.route)std::cin>>v;return c;};
   int has_pending;std::cin>>has_pending;std::optional<Chase> pending;if(has_pending)pending=read_chase();
   int threat_count;std::cin>>threat_count;std::vector<Chase> threats;for(int i=0;i<threat_count;++i)threats.push_back(read_chase());
   std::cin>>k;std::vector<int> legal(k);for(int& m:legal)std::cin>>m;
   int move,reveal,accepted;std::cin>>move>>reveal>>accepted;
   if(!std::cin)throw std::runtime_error("truncated corpus");
   auto actual=b.moves();std::sort(actual.begin(),actual.end());std::sort(legal.begin(),legal.end());
   double v=model->value(a),fresh=model->value(model->init(f));
   max_error=std::max(max_error,std::abs(v-expected));max_increment_error=std::max(max_increment_error,std::abs(v-fresh));
   if(b.turn!=turn||b.winner!=winner||b.draw!=bool(draw)||b.quiet!=quiet||b.repetition()!=rep||b.grid!=g||b.token!=tokens||
      b.remaining!=remaining||b.captured!=cap||b.pending!=pending||b.threats!=threats||actual!=legal||std::abs(v-expected)>1e-5||std::abs(v-fresh)>1e-6){
    std::cerr<<"parity failure game "<<game<<" ply "<<ply<<" NNUE "<<v<<" expected "<<expected<<" moves "<<actual.size()<<" vs "<<legal.size()<<"\n";return 1;
   }
   ++positions;terminal+=b.over();chase+=bool(b.pending);repeated+=b.repetition()>1;
   if(move<0)continue;
   Board prior=b;
   if(b.play(move,reveal)!=bool(accepted))return 1;
   if(!accepted&&!(b==prior))return 1;
   auto next=features(b);a=model->advance(a,f,next);f=next;
  }
 }
 if(!terminal||!chase||!repeated)return 1;
 std::cout<<std::setprecision(12)<<"{\"games\":"<<games<<",\"positions\":"<<positions<<",\"terminal_positions\":"<<terminal<<",\"pending_chase_positions\":"<<chase<<",\"repeated_positions\":"<<repeated<<",\"max_nnue_error\":"<<max_error<<",\"max_increment_error\":"<<max_increment_error<<",\"parity\":true}\n";
}
