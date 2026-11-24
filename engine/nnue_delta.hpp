#pragma once
#include "nnue.hpp"
namespace dc {
// Single legal public move; no hidden identity. Preserve ascending feature order.
inline Accumulator advance_move(const NNUE& model,const Accumulator& before,const Board& old,const Board& next,int move){
 Accumulator a=before;
 auto add=[&](int feature,double delta){
  if(delta!=0)for(int i=0;i<256;++i)a[i]+=double(model.columns[feature][i])*delta;
 };
 int from=move/32,to=move%32;
 auto square=[&](int s){
  int p=old.grid[s],q=next.grid[s];if(p==q)return;
  if(p>0&&q>0&&q<p){add(s*15+q-1,1);add(s*15+p-1,-1);}
  else{if(p>0)add(s*15+p-1,-1);if(q>0)add(s*15+q-1,1);}
 };
 square(std::min(from,to));if(from!=to)square(std::max(from,to));
 if(from==to){
  int p=next.grid[from]-1;
  add(480+p,double(float(next.remaining[p])/inventory[p])-float(old.remaining[p])/inventory[p]);
 }
 for(int c=0;c<2;++c)add(494+c,int(next.turn==c)-int(old.turn==c));
 add(496,double(std::min(float(next.quiet)/60,1.f))-std::min(float(old.quiet)/60,1.f));
 add(497,double(std::min(float(next.repetition())/3,1.f))-std::min(float(old.repetition())/3,1.f));
 return a;
}
}
