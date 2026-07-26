import React, { useState } from 'react';
import { Board as EngineBoard, Piece } from '../engine/board';

interface BoardProps {
  engine: EngineBoard;
  onMove?: () => void;
}

const pieceLabels: Record<Piece, string> = {
  [Piece.EMPTY]: '',
  [Piece.RED_KING]: '帥',
  [Piece.RED_GUARD]: '仕',
  [Piece.RED_MINISTER]: '相',
  [Piece.RED_ROOK]: '俥',
  [Piece.RED_KNIGHT]: '傌',
  [Piece.RED_CANNON]: '炮',
  [Piece.RED_PAWN]: '兵',
  [Piece.BLK_KING]: '將',
  [Piece.BLK_GUARD]: '士',
  [Piece.BLK_MINISTER]: '象',
  [Piece.BLK_ROOK]: '車',
  [Piece.BLK_KNIGHT]: '馬',
  [Piece.BLK_CANNON]: '包',
  [Piece.BLK_PAWN]: '卒',
  [Piece.HIDDEN]: '?',
};

export const Board: React.FC<BoardProps> = ({ engine, onMove }) => {
  const [trigger, setTrigger] = useState(0);

  const handleClick = (r: number, c: number) => {
    try {
      if (engine.getPiece(r, c) === Piece.HIDDEN) {
        engine.flipPiece(r, c);
        setTrigger(trigger + 1);
        if (onMove) onMove();
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="grid grid-cols-8 gap-2 p-6 bg-slate-800/50 rounded-2xl shadow-2xl backdrop-blur-md border border-slate-700">
      {engine.grid.map((row, r) =>
        row.map((piece, c) => {
          const isHidden = piece === Piece.HIDDEN;
          const isRed = piece >= Piece.RED_KING && piece <= Piece.RED_PAWN;
          const isBlack = piece >= Piece.BLK_KING && piece <= Piece.BLK_PAWN;

          return (
            <div
              key={`${r}-${c}`}
              onClick={() => handleClick(r, c)}
              className={`
                flex items-center justify-center w-12 h-12 rounded-full cursor-pointer select-none
                transition-all duration-300 hover:scale-110 shadow-lg border-2
                ${isHidden ? 'bg-slate-700 border-slate-600 text-transparent hover:bg-slate-600' : ''}
                ${isRed ? 'bg-slate-800 border-red-500 text-red-500 shadow-red-500/20' : ''}
                ${isBlack ? 'bg-slate-800 border-emerald-400 text-emerald-400 shadow-emerald-400/20' : ''}
                ${piece === Piece.EMPTY ? 'opacity-0 cursor-default' : ''}
              `}
            >
              <span className={`text-xl font-bold ${isHidden ? 'hidden' : 'block'}`}>
                {pieceLabels[piece]}
              </span>
            </div>
          );
        })
      )}
    </div>
  );
};
