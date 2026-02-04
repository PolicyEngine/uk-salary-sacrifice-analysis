import React from "react";
import { formatCap } from "../utils/formatters";

export default function CapSelector({ cap, onChange }) {
  return (
    <>
      <div className="cap-slider-value">{formatCap(cap)}</div>
      <input
        className="cap-slider"
        type="range"
        min={0}
        max={10000}
        step={1000}
        value={cap}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="cap-slider-labels">
        <span>{formatCap(0)}</span>
        <span>{formatCap(10000)}</span>
      </div>
    </>
  );
}
