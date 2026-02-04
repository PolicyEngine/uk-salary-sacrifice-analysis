import React from "react";

export default function DisplayToggle({ display, onChange }) {
  return (
    <div className="control-group">
      <h3>Display</h3>
      <div className="toggle-buttons">
        <button
          className={`toggle-btn${display === "relative" ? " active" : ""}`}
          onClick={() => onChange("relative")}
        >
          Relative (%)
        </button>
        <button
          className={`toggle-btn${display === "absolute" ? " active" : ""}`}
          onClick={() => onChange("absolute")}
        >
          Absolute (£)
        </button>
      </div>
    </div>
  );
}
