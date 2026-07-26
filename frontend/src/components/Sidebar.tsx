import React from 'react';

export const Sidebar: React.FC = () => {
  return (
    <div className="flex flex-col gap-6 p-6 w-80 bg-slate-800/80 rounded-2xl shadow-2xl backdrop-blur-md border border-slate-700 text-slate-200">
      <div className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
        DarkChess NNUE
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Eval Bar (勝率預測)</label>
        <div className="h-4 w-full bg-slate-700 rounded-full overflow-hidden relative shadow-inner">
          <div className="absolute top-0 left-0 h-full bg-gradient-to-r from-red-500 to-red-400 transition-all duration-500 ease-in-out w-1/2"></div>
          <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold mix-blend-difference text-white">
            0.00
          </div>
        </div>
        <div className="flex justify-between text-xs font-medium text-slate-500">
          <span className="text-red-400">紅方</span>
          <span className="text-emerald-400">黑方</span>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Opponent (對手)</label>
        <select className="bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg focus:ring-emerald-500 focus:border-emerald-500 block w-full p-2.5 outline-none transition-colors">
          <option value="latest">Nightly Build (Latest NNUE)</option>
          <option value="gen1">Gen 1 (Heuristic)</option>
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Time Control (時間限制)</label>
        <select className="bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg focus:ring-emerald-500 focus:border-emerald-500 block w-full p-2.5 outline-none transition-colors">
          <option value="1">1 秒 (快棋)</option>
          <option value="5">5 秒 (標準)</option>
          <option value="0">無限制 (最強)</option>
        </select>
      </div>

      <div className="mt-auto pt-6 border-t border-slate-700 flex items-center gap-3">
        <label className="relative inline-flex items-center cursor-pointer">
          <input type="checkbox" className="sr-only peer" />
          <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500 shadow-inner"></div>
        </label>
        <span className="text-sm font-medium text-slate-300">協助背景訓練 (Self-Play)</span>
      </div>
    </div>
  );
};
