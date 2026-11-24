#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using U=uint32_t;
struct Position { std::array<int,32> g{}; int turn=2; bool terminal=false; };
struct Geometry {
 std::array<U,32> near{}, lines{};
 std::array<std::array<U,32>,32> between{};
 Geometry() {
  for(int a=0;a<32;++a) for(int b=0;b<32;++b) if(a!=b){
   int ar=a/8,ac=a%8,br=b/8,bc=b%8;
   if((ar==br && std::abs(ac-bc)==1)||(ac==bc && std::abs(ar-br)==1)) near[a]|=U(1)<<b;
   if(ar!=br && ac!=bc) continue;
   lines[a]|=U(1)<<b;
   int step=ar==br?(a<b?1:-1):(a<b?8:-8);
   for(int x=a+step;x!=b;x+=step) between[a][b]|=U(1)<<x;
  }
 }
};
const Geometry geo;
int color(int p){return p>=1&&p<=7?0:p>=8&&p<=14?1:2;}
int type(int p){return (p-1)%7;}
struct Bits {
 U empty=0,hidden=0,red=0,black=0,occupied=0;
 explicit Bits(const Position& p){
  for(int i=0;i<32;++i){ U b=U(1)<<i; int q=p.g[i]; if(q==0)empty|=b;else if(q==15)hidden|=b;else if(color(q)==0)red|=b;else black|=b;}
  occupied=~empty;
 }
};
template<class Emit> void moves(const Position& p,Emit emit){
 if(p.terminal)return;
 Bits b(p);
 for(U x=b.hidden;x;x&=x-1){int i=std::countr_zero(x);emit(i*32+i);}
 if(p.turn==2)return;
 U own=p.turn==0?b.red:b.black, enemy=p.turn==0?b.black:b.red;
 for(U x=own;x;x&=x-1){
  int a=std::countr_zero(x),t=type(p.g[a]);
  for(U y=geo.near[a]&b.empty;y;y&=y-1)emit(a*32+std::countr_zero(y));
  for(U y=(t==5?geo.lines[a]:geo.near[a])&enemy;y;y&=y-1){
   int dst=std::countr_zero(y),v=type(p.g[dst]);
   bool legal=t==5?std::popcount(geo.between[a][dst]&b.occupied)==1:
      t==0&&v==6?false:t==6&&v==0?true:t<=v;
   if(legal)emit(a*32+dst);
  }
 }
}
int main(){
 int n=0,rounds=0;if(!(std::cin>>n>>rounds)||n<=0||rounds<=0)return 2;
 std::vector<Position> corpus(n); uint64_t expected_checksum=0;
 for(int i=0;i<n;++i){
  auto& p=corpus[i];int terminal,k;
  if(!(std::cin>>p.turn>>terminal))return 2;p.terminal=terminal;
  for(int& v:p.g)if(!(std::cin>>v)||v<0||v>15)return 2;
  if(!(std::cin>>k)||k<0||k>1024)return 2;
  std::vector<int> expected(k),actual;
  for(int& v:expected)if(!(std::cin>>v))return 2;
  moves(p,[&](int m){actual.push_back(m);});
  std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
  if(actual!=expected){std::cerr<<"Parity mismatch "<<i<<"\n";return 1;}
  for(int m:actual)expected_checksum+=m+1;
 }
 uint64_t checksum=0;
 auto start=std::chrono::steady_clock::now();
 for(int j=0;j<rounds;++j)for(const auto& p:corpus)moves(p,[&](int m){checksum+=m+1;});
 double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 if(checksum!=expected_checksum*uint64_t(rounds))return 1;
 std::cout<<"{\"positions\":"<<n<<",\"rounds\":"<<rounds<<",\"seconds\":"<<s<<",\"positions_per_second\":"<<(n*double(rounds)/s)<<",\"checksum\":"<<checksum<<",\"parity\":true}\n";
}
