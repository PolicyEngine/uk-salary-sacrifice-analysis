import React from "react";
import { YEAR_LABELS } from "../config/constants";
import InfoTooltip from "./InfoTooltip";

export default function YearToggle({ year, onChange }) {
  return (
    <div className="control-group">
      <h3>
        Tax year
        <InfoTooltip
          title="Tax year"
          description="The fiscal year for the analysis. The salary sacrifice cap takes effect from April 2029. Select 2030-31 to see projections for the following year."
        />
      </h3>
      <select
        className="control-select"
        value={year}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {Object.entries(YEAR_LABELS).map(([key, label]) => (
          <option key={key} value={key}>
            {label}
          </option>
        ))}
      </select>
    </div>
  );
}
