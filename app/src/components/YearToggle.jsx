import React from "react";
import { YEAR_LABELS } from "../config/constants";

export default function YearToggle({ year, onChange }) {
  return (
    <div className="control-group">
      <h3>Tax year</h3>
      <div className="toggle-buttons">
        {Object.entries(YEAR_LABELS).map(([key, label]) => (
          <button
            key={key}
            className={`toggle-btn${year === Number(key) ? " active" : ""}`}
            onClick={() => onChange(Number(key))}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
