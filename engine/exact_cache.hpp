#pragma once
#include "board.hpp"
#include <unordered_map>
namespace dc {
// Only irreversible-boundary states: no history or chase state is omitted.
// Compare full keys; a hash collision cannot return the wrong value.
struct ExactKey {
 uint64_t lo=0,hi=0,meta=0;
 bool operator==(const ExactKey&)const=default;
};
struct ExactKeyHash {
 size_t operator()(const ExactKey& k)const noexcept{
  auto mix=[](uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);};
  return size_t(mix(k.lo)^std::rotl(mix(k.hi),21)^std::rotl(mix(k.meta),42));
 }
};
inline bool cacheable(const Board& b,int depth){
 return depth>0&&depth<=12&&b.turn!=NONE&&!b.over()&&b.quiet==0&&b.reversible.empty()&&b.threats.empty()&&!b.pending;
}
inline ExactKey exact_key(const Board& b,int depth){
 ExactKey k;
 for(int i=0;i<16;++i){k.lo|=uint64_t(b.grid[i])<<(4*i);k.hi|=uint64_t(b.grid[i+16])<<(4*i);}
 for(int i=0;i<14;++i)k.meta|=uint64_t(b.remaining[i])<<(3*i);
 k.meta|=uint64_t(b.turn)<<42;k.meta|=uint64_t(depth)<<44;
 return k;
}
using ExactCache=std::unordered_map<ExactKey,double,ExactKeyHash>;
}
