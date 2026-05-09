'use client';

import React, { useState, useEffect } from "react";
import { useSimulationData } from "./hooks/useSimulationData";
import {
  SCENARIO_LABELS,
  YEAR_LABELS,
  BASELINE_LABELS,
} from "./config/constants";
import ScenarioSelector from "./components/ScenarioSelector";
import YearToggle from "./components/YearToggle";
import DisplayToggle from "./components/DisplayToggle";
import BaselineToggle from "./components/BaselineToggle";
import CapSelector from "./components/CapSelector";
import RevenueByCapChart from "./components/RevenueByCapChart";
import RevenueSummary from "./components/RevenueSummary";
import DecileChart from "./components/DecileChart";
import AffectedShareChart from "./components/AffectedShareChart";
import InfoTooltip from "./components/InfoTooltip";
import { formatCap } from "./utils/formatters";

// Read query parameters from the URL during SSR-safe initial state hydration.
// Returns defaults if window is unavailable (App Router prerender pass).
function getInitialState() {
  const search =
    typeof window === "undefined" ? "" : window.location.search;
  const params = new URLSearchParams(search);
  return {
    cap: Number(params.get("cap")) || 5000,
    year: Number(params.get("year")) || 2029,
    employer: params.get("employer") || "spread",
    employee: params.get("employee") || "maintain",
    display: params.get("display") || "absolute",
    baseline: params.get("baseline") || "2000",
  };
}

function syncQueryParams(state) {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams();
  params.set("cap", state.cap);
  params.set("year", state.year);
  params.set("employer", state.employer);
  params.set("employee", state.employee);
  params.set("display", state.display);
  params.set("baseline", state.baseline);
  const newUrl =
    window.location.pathname + "?" + params.toString() + window.location.hash;
  window.history.replaceState(null, "", newUrl);
}

function BehavioralTab({
  data,
  cap,
  setCap,
  year,
  setYear,
  employer,
  setEmployer,
  employee,
  setEmployee,
  display,
  setDisplay,
  baseline,
  setBaseline,
}) {
  const scenario = `${employer}_${employee === "cash" ? "cash" : "maintain"}`;

  const baselineLabel =
    baseline === "none" ? "no cap" : `${BASELINE_LABELS[baseline]}`;

  return (
    <>
      <p className="intro-text">
        Salary sacrifice arrangements allow employees to exchange part of their
        salary for non-cash benefits before tax and National Insurance are
        calculated. The government announced a &pound;2,000 annual cap on
        NI-exempt pension salary sacrifice contributions in the{" "}
        <a href="https://obr.uk/efo/economic-and-fiscal-outlook-october-2025/">
          Autumn Budget 2025
        </a>
        , taking effect from April 2029. Above this threshold, standard NI rates
        would apply. This tool models revenue and distributional outcomes under
        different behavioural assumptions about how employers and employees
        might adjust their contributions in response to the cap. See the{" "}
        <a href="https://www.policyengine.org/uk/research/uk-salary-sacrifice-cap">
          full report
        </a>{" "}
        for methodology.
      </p>
      <div className="controls-card">
        <div className="controls-card-header">
          <h2>Analysis settings</h2>
        </div>
        <div className="controls-section">
          <div className="controls-section-label">Policy</div>
          <div className="controls-row">
            <BaselineToggle baseline={baseline} onChange={setBaseline} />
            <div className="control-group">
              <h3>
                Cap level
                <InfoTooltip
                  title="Cap level"
                  description="The maximum annual amount that can be contributed through salary sacrifice without paying National Insurance. Contributions above this cap are subject to standard NI rates."
                />
              </h3>
              <CapSelector cap={cap} onChange={setCap} />
            </div>
            <YearToggle year={year} onChange={setYear} />
          </div>
        </div>
        <div className="controls-section">
          <div className="controls-section-label">Behavioural assumptions</div>
          <div className="controls-row">
            <ScenarioSelector
              employer={employer}
              employee={employee}
              onEmployerChange={setEmployer}
              onEmployeeChange={setEmployee}
            />
          </div>
        </div>
      </div>
      <div className="chart-section">
        <h2>Revenue by cap level</h2>
        <p className="section-description">
          This chart shows estimated government revenue at each cap level under
          the selected behavioural scenario. The highlighted bar indicates the
          currently selected cap.
        </p>
        <div className="chart-container" style={{ height: 350 }}>
          <RevenueByCapChart
            data={data.results}
            year={year}
            scenario={scenario}
            cap={cap}
            baseline={baseline}
          />
        </div>
      </div>
      <div className="chart-section">
        <h2>Revenue by scenario at {formatCap(cap)} cap</h2>
        <p className="section-description">
          This table compares revenue across the four behavioural scenarios at
          the selected cap level: each combination of employer response (spread
          or absorb cost) and employee response (maintain pension or take cash).
        </p>
        <RevenueSummary
          data={data.results}
          cap={cap}
          year={year}
          baseline={baseline}
        />
      </div>
      <div className="chart-grid">
        <div className="chart-section">
          <h2>Distributional impact by income decile</h2>
          <p className="section-description">
            This chart shows the average change in household disposable income
            for each income decile under the selected scenario and cap level.
          </p>
          <DisplayToggle display={display} onChange={setDisplay} />
          <div className="chart-container">
            <DecileChart
              data={data.results}
              cap={cap}
              year={year}
              scenario={scenario}
              display={display}
              baseline={baseline}
            />
          </div>
        </div>
        <div className="chart-section">
          <h2>Winners and losers by income decile</h2>
          <p className="section-description">
            This chart shows the percentage of people in each income decile who
            gain, lose, or are unaffected by the cap under the selected scenario.
          </p>
          <div className="chart-container">
            <AffectedShareChart
              data={data.results}
              cap={cap}
              year={year}
              scenario={scenario}
              baseline={baseline}
            />
          </div>
        </div>
      </div>
    </>
  );
}

export default function App() {
  const { data, loading, error } = useSimulationData();

  const initial = getInitialState();
  const [cap, setCap] = useState(initial.cap);
  const [year, setYear] = useState(initial.year);
  const [employer, setEmployer] = useState(initial.employer);
  const [employee, setEmployee] = useState(initial.employee);
  const [display, setDisplay] = useState(initial.display);
  const [baseline, setBaseline] = useState(initial.baseline);

  useEffect(() => {
    syncQueryParams({ cap, year, employer, employee, display, baseline });
  }, [cap, year, employer, employee, display, baseline]);

  if (loading) {
    return (
      <div className="loading-state">
        <p>Loading simulation data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <p>Failed to load data: {error}</p>
      </div>
    );
  }

  return (
    <>
      <header className="app-header">
        <div className="header-text">
          <h1>Salary sacrifice cap analysis tool</h1>
        </div>
      </header>
      <div className="app-container">
        {data && (
          <BehavioralTab
            data={data}
            cap={cap}
            setCap={setCap}
            year={year}
            setYear={setYear}
            employer={employer}
            setEmployer={setEmployer}
            employee={employee}
            setEmployee={setEmployee}
            display={display}
            setDisplay={setDisplay}
            baseline={baseline}
            setBaseline={setBaseline}
          />
        )}
      </div>
      <footer className="app-footer">
        Analysis by <a href="https://policyengine.org">PolicyEngine</a> |{" "}
        <a href="https://github.com/PolicyEngine/uk-salary-sacrifice-analysis">
          Source code
        </a>
      </footer>
    </>
  );
}
