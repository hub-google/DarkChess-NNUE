#pragma once
#include "board.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace dc {
using Features=std::array<float,498>;
using Accumulator=std::array<double,256>;
inline Features features(const Board& b){
 Features f{};
 for(int i=0;i<32;++i)if(b.grid[i]>0)f[i*15+b.grid[i]-1]=1;
 for(int i=0;i<14;++i)f[480+i]=float(b.remaining[i])/inventory[i];
 if(b.turn!=NONE)f[494+b.turn]=1;
 f[496]=std::min(float(b.quiet)/60,1.f);f[497]=std::min(float(b.repetition())/3,1.f);return f;
}
// Fixed v1 portable format, same tensor order as validated browser manifest.
// Caller must verify manifest and binary SHA-256 before constructing the model.
struct NNUE {
 std::array<std::array<float,256>,498> columns{};
 std::array<float,256> bias{};
 std::array<std::array<float,256>,32> w2{};
 std::array<float,32> b2{},w3{};
 float b3=0;
 explicit NNUE(const char* path){
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("model open");
  auto read=[&](){
   std::array<unsigned char,4> b{};if(!in.read(reinterpret_cast<char*>(b.data()),4))throw std::runtime_error("model truncated");
   uint32_t n=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
   float v=std::bit_cast<float>(n);if(!std::isfinite(v))throw std::runtime_error("nonfinite model");return v;
  };
  for(int i=0;i<256;++i)for(auto& col:columns)col[i]=read();
  for(auto& v:bias)v=read();
  for(auto& row:w2)for(auto& v:row)v=read();
  for(auto& v:b2)v=read();
  for(auto& v:w3)v=read();
  b3=read();if(in.peek()!=EOF)throw std::runtime_error("unexpected model length");
 }
 Accumulator init(const Features& f)const{
  Accumulator a{};std::copy(bias.begin(),bias.end(),a.begin());
  for(int col=0;col<498;++col)if(f[col]!=0)for(int i=0;i<256;++i)a[i]+=double(columns[col][i])*f[col];
  return a;
 }
 Accumulator advance(const Accumulator& a,const Features& before,const Features& after)const{
  Accumulator next=a;
  for(int c=0;c<498;++c)if(before[c]!=after[c]){
   double delta=double(after[c])-before[c];
   for(int i=0;i<256;++i)next[i]+=double(columns[c][i])*delta;
  }
  return next;
 }
 double value(const Accumulator& a)const{
  std::array<float,256> h1{};
  for(int i=0;i<256;++i)h1[i]=float(std::clamp(a[i],0.,1.));
  double out=b3;
  for(int i=0;i<32;++i){
   double sum=b2[i];for(int j=0;j<256;++j)sum+=double(w2[i][j])*h1[j];
   float h2=float(std::clamp(sum,0.,1.));out+=double(w3[i])*h2;
  }
  return std::tanh(out);
 }
};
}
