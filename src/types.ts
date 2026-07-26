export enum Color {
  Red = 0,
  Black = 1,
  None = 2,
}

export enum PieceType {
  Empty = 0,
  Pawn = 1, // 兵/卒
  Cannon = 2, // 炮/包
  Knight = 3, // 傌/馬
  Rook = 4, // 俥/車
  Elephant = 5, // 相/象
  Advisor = 6, // 仕/士
  King = 7, // 帥/將
}

export type Piece = number;

export const makePiece = (color: Color, type: PieceType): Piece => {
  if (type === PieceType.Empty) return 0;
  return (color << 3) | type;
};

export const getPieceColor = (piece: Piece): Color => {
  if (piece === PieceType.Empty) return Color.None;
  return piece >> 3;
};

export const getPieceType = (piece: Piece): PieceType => {
  return piece & 0b111;
};

// 棋盤為 4x8 = 32 格，以一維陣列 0~31 表示
export type Square = number;

// Move 結構：使用一個 16-bit 整數打包 Move 資訊
// 0-4 bits: from (0-31)
// 5-9 bits: to (0-31)
// 10-11 bits: move type (0: move, 1: capture, 2: flip)
export type Move = number;

export enum MoveType {
  Move = 0,
  Capture = 1,
  Flip = 2,
}

export const makeMove = (from: Square, to: Square, type: MoveType): Move => {
  return from | (to << 5) | (type << 10);
};

export const getFrom = (move: Move): Square => move & 0x1f;
export const getTo = (move: Move): Square => (move >> 5) & 0x1f;
export const getMoveType = (move: Move): MoveType => (move >> 10) & 0x3;
