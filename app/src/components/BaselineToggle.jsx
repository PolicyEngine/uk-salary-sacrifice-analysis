import React from "react";
import { BASELINE_LABELS } from "../config/constants";
import InfoTooltip from "./InfoTooltip";

export default function BaselineToggle({ baseline, onChange }) {
  return (
    <div className="control-group">
      <h3>
        Baseline
        <InfoTooltip
          title="Baseline"
          description="The reference policy to compare against. '£2k cap' uses the cap announced in the Autumn Budget 2025 (current law). 'No cap' uses pre-budget policy with no limit on salary sacrifice."
        />
      </h3>
      <select
        className="control-select"
        value={baseline}
        onChange={(e) => onChange(e.target.value)}
      >
        {Object.entries(BASELINE_LABELS).map(([key, label]) => (
          <option key={key} value={key}>
            {label}
          </option>
        ))}
      </select>
    </div>
  );
}
