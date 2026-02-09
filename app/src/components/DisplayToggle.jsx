import React from "react";

export default function DisplayToggle({ display, onChange }) {
  return (
    <div className="control-group">
      <select
        className="control-select control-select-inline"
        value={display}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="absolute">Absolute (£)</option>
        <option value="relative">Relative (%)</option>
      </select>
    </div>
  );
}
