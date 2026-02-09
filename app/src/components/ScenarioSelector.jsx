import React from "react";
import { EMPLOYER_LABELS, EMPLOYEE_LABELS } from "../config/constants";
import InfoTooltip from "./InfoTooltip";

export default function ScenarioSelector({
  employer,
  employee,
  onEmployerChange,
  onEmployeeChange,
}) {
  return (
    <>
      <div className="control-group">
        <h3>
          Employer response
          <InfoTooltip
            title="Employer response"
            description="How employers handle increased NI costs. 'Spread cost' distributes costs across all employees as a small pay reduction. 'Absorb cost' means the employer bears the full cost."
          />
        </h3>
        <select
          className="control-select"
          value={employer}
          onChange={(e) => onEmployerChange(e.target.value)}
        >
          {Object.entries(EMPLOYER_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <div className="control-group">
        <h3>
          Employee response
          <InfoTooltip
            title="Employee response"
            description="How employees redirect contributions above the cap. 'Maintain pension' shifts the excess to regular employee pension contributions. 'Take cash' means the excess is taken as taxable salary."
          />
        </h3>
        <select
          className="control-select"
          value={employee}
          onChange={(e) => onEmployeeChange(e.target.value)}
        >
          {Object.entries(EMPLOYEE_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </div>
    </>
  );
}
