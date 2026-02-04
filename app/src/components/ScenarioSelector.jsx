import React from "react";
import { EMPLOYER_LABELS, EMPLOYEE_LABELS } from "../config/constants";

export default function ScenarioSelector({
  employer,
  employee,
  onEmployerChange,
  onEmployeeChange,
}) {
  return (
    <>
      <div className="control-group">
        <h3>Employer response</h3>
        <div className="toggle-buttons">
          {Object.entries(EMPLOYER_LABELS).map(([key, label]) => (
            <button
              key={key}
              className={`toggle-btn${employer === key ? " active" : ""}`}
              onClick={() => onEmployerChange(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="control-group">
        <h3>Employee response</h3>
        <div className="toggle-buttons">
          {Object.entries(EMPLOYEE_LABELS).map(([key, label]) => (
            <button
              key={key}
              className={`toggle-btn${employee === key ? " active" : ""}`}
              onClick={() => onEmployeeChange(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}
