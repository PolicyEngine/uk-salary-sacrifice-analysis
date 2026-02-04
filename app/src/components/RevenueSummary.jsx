import React from "react";
import { SCENARIO_KEYS, SCENARIO_LABELS } from "../config/constants";
import { formatBillions } from "../utils/formatters";
import { getAllScenariosRevenue } from "../utils/dataTransformers";

export default function RevenueSummary({ data, cap, year, baseline }) {
  const revenues = getAllScenariosRevenue(
    data,
    cap,
    year,
    SCENARIO_KEYS,
    baseline,
  );

  return (
    <div className="revenue-grid">
      {SCENARIO_KEYS.map((key) => (
        <div key={key} className="revenue-card">
          <div className="label">{SCENARIO_LABELS[key]}</div>
          <div className="value">
            {revenues[key] != null ? formatBillions(revenues[key]) : "\u2014"}
          </div>
        </div>
      ))}
    </div>
  );
}
