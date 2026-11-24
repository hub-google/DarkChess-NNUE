#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <vector>

namespace dc {
using Mask=uint32_t;
using Grid=std::array<int,32>;
constexpr int RED=0,BLACK=1,NONE=2,EMPTY=0,HIDDEN=15;
constexpr std::array<int,14> inventory{1,2,2,2,2,2,5,1,2,2,2,2,2,5};
inline int color(int p){return p>=1&&p<=7?RED:p>=8&&p<=14?BLACK:NONE;}
inline int kind(int p){return (p-1)%7;}
inline Mask bit(int s){return Mask(1)<<s;}
struct Geometry {
 std::array<Mask,32> adjacent{},line{};
 std::array<std::array<Mask,32>,32> between{};
 Geometry(){
  for(int a=0;a<32;++a)for(int b=0;b<32;++b)if(a!=b){
   int ar=a/8,ac=a%8,br=b/8,bc=b%8;
   if((ar==br&&std::abs(ac-bc)==1)||(ac==bc&&std::abs(ar-br)==1))adjacent[a]|=bit(b);
   if(ar!=br&&ac!=bc)continue;
   line[a]|=bit(b);
   int step=ar==br?(a<b?1:-1):(a<b?8:-8);
   for(int x=a+step;x!=b;x+=step)between[a][b]|=bit(x);
  }
 }
};
inline const Geometry geometry{};
struct Chase {
 int attacker,target,count,chaser;
 std::vector<int> route;
 bool operator==(const Chase&)const=default;
};
struct Snapshot {
 Grid grid;
 std::array<int,14> remaining;
 int turn;
 bool operator==(const Snapshot&)const=default;
};
// Public information only. A referee supplies explicit reveal outcomes.
struct Board {
 Grid grid{},token{};
 std::array<int,14> remaining=inventory;
 int turn=NONE,winner=NONE,ply=0,quiet=0;
 bool draw=false;
 std::vector<Snapshot> reversible;
 std::vector<Chase> threats;
 std::optional<Chase> pending;
 std::vector<int> captured;
 Board(){grid.fill(HIDDEN);for(int i=0;i<32;++i)token[i]=i;}
 bool over()const{return winner!=NONE||draw;}
 bool operator==(const Board&)const=default;
 Snapshot snapshot()const{return {grid,remaining,turn};}
 int repetition()const{
  if(!quiet)return 1;
  return 1+int(std::count(reversible.begin(),reversible.end(),snapshot()));
 }
 int square_of(int t)const{
  auto it=std::find(token.begin(),token.end(),t);
  return it==token.end()?-1:int(it-token.begin());
 }
 static Mask occupancy(const Grid& g){
  Mask x=0;for(int i=0;i<32;++i)if(g[i]!=EMPTY)x|=bit(i);return x;
 }
 static bool attacks(int a,int b,const Grid& g,Mask occ){
  if(a<0||b<0||a==b||color(g[a])==NONE||color(g[b])==NONE||color(g[a])==color(g[b]))return false;
  int t=kind(g[a]),v=kind(g[b]);
  if(t==5)return (geometry.line[a]&bit(b))&&std::popcount(geometry.between[a][b]&occ)==1;
  return (geometry.adjacent[a]&bit(b))&&(t==0&&v==6?false:t==6&&v==0?true:t<=v);
 }
 bool forbidden(int a,int b)const{
  if(!pending||token[a]!=pending->attacker||grid[b]!=EMPTY)return false;
  int target=square_of(pending->target);if(target<0)return false;
  Grid next=grid;next[b]=next[a];next[a]=EMPTY;
  if(!attacks(b,target,next,occupancy(next)))return false;
  Mask visited=0;for(int s:pending->route){if(visited&bit(s))return true;visited|=bit(s);}return false;
 }
 template<class Emit> void for_moves(Emit emit)const{
  if(over())return;
  Mask hidden=0,empty=0,red=0,black=0;
  for(int i=0;i<32;++i){int p=grid[i];if(p==HIDDEN)hidden|=bit(i);else if(p==EMPTY)empty|=bit(i);else if(color(p)==RED)red|=bit(i);else black|=bit(i);}
  for(Mask x=hidden;x;x&=x-1){int s=std::countr_zero(x);emit(s*32+s);}
  if(turn==NONE)return;
  Mask own=turn==RED?red:black,enemy=turn==RED?black:red;
  for(Mask x=own;x;x&=x-1){
   int a=std::countr_zero(x),t=kind(grid[a]);
   Mask candidates=(geometry.adjacent[a]&empty)|((t==5?geometry.line[a]:geometry.adjacent[a])&enemy);
   for(Mask y=candidates;y;y&=y-1){
    int b=std::countr_zero(y);
    if(grid[b]==EMPTY?!forbidden(a,b):attacks(a,b,grid,~empty))emit(a*32+b);
   }
  }
 }
 std::vector<int> moves()const{std::vector<int> out;out.reserve(64);for_moves([&](int m){out.push_back(m);});return out;}
 bool legal(int m)const{
  if(m<0||m>=1024||over())return false;
  int a=m/32,b=m%32;
  if(a==b)return grid[a]==HIDDEN;
  if(turn==NONE||color(grid[a])!=turn)return false;
  if(grid[b]==EMPTY)return (geometry.adjacent[a]&bit(b))&&!forbidden(a,b);
  return attacks(a,b,grid,occupancy(grid));
 }
 void update_chase(int mover,int moved,bool irreversible){
  auto old=std::move(threats);auto old_pending=std::move(pending);
  threats.clear();pending.reset();if(irreversible)return;
  int square=square_of(moved);if(square<0)return;
  Mask occ=occupancy(grid);
  for(const auto& t:old){
   if(moved!=t.target||mover==t.chaser)continue;
   int a=square_of(t.attacker);
   if(a>=0&&!attacks(a,square,grid,occ)){pending=t;pending->route.push_back(square);return;}
  }
  for(int b=0;b<32;++b)if(attacks(square,b,grid,occ)){
   int target=token[b];bool same=old_pending&&old_pending->attacker==moved&&old_pending->target==target&&old_pending->chaser==mover;
   threats.push_back({moved,target,same?old_pending->count+1:1,mover,same?old_pending->route:std::vector<int>{b}});
  }
 }
 // Reveal is mandatory for flips and forbidden for deterministic moves.
 // Invalid calls do not mutate any state.
 bool play(int m,int reveal=0){
  if(!legal(m))return false;
  int a=m/32,b=m%32,mover=turn,moved=token[a];bool flip=a==b;
  if(flip?(reveal<1||reveal>14||remaining[reveal-1]<=0):reveal!=0)return false;
  bool capture=!flip&&grid[b]!=EMPTY;
  if(flip||capture){quiet=0;reversible.clear();}
  else {reversible.push_back(snapshot());++quiet;}
  if(flip){grid[a]=reveal;--remaining[reveal-1];if(turn==NONE)turn=color(reveal);}
  else {if(capture)captured.push_back(grid[b]);grid[b]=grid[a];grid[a]=EMPTY;token[b]=moved;token[a]=-1;}
  update_chase(mover,moved,flip||capture);turn=1-turn;++ply;
  bool red=false,black=false,hidden=false;
  for(int p:grid){red|=color(p)==RED;black|=color(p)==BLACK;hidden|=p==HIDDEN;}
  for(int i=0;i<14;++i)if(remaining[i]){if(i<7)red=true;else black=true;}
  if(!red)winner=BLACK;
  else if(!black)winner=RED;
  else if(quiet>=60)draw=true;
  else if(!hidden&&moves().empty())winner=1-turn;
  return true;
 }
};
}
