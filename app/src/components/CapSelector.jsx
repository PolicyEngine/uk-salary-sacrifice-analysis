import React from "react";
import { CAPS } from "../config/constants";
import { formatCap } from "../utils/formatters";

export default function CapSelector({ cap, onChange }) {
  return (
    <select
      className="control-select"
      value={cap}
      onChange={(e) => onChange(Number(e.target.value))}
    >
      {CAPS.map((c) => (
        <option key={c} value={c}>
          {formatCap(c)}
        </option>
      ))}
    </select>
  );
}
