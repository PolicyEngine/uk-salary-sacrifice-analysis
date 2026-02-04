import React from "react";
import { BASELINE_LABELS } from "../config/constants";

export default function BaselineToggle({ baseline, onChange }) {
  return (
    <div className="control-group">
      <h3>Baseline</h3>
      <div className="toggle-buttons">
        {Object.entries(BASELINE_LABELS).map(([key, label]) => (
          <button
            key={key}
            className={`toggle-btn${baseline === key ? " active" : ""}`}
            onClick={() => onChange(key)}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
