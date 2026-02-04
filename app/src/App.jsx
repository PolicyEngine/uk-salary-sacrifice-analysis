import React, { useState, useEffect, useMemo } from "react";
import { useSimulationData } from "./hooks/useSimulationData";
import {
  SCENARIO_LABELS,
  YEAR_LABELS,
  BASELINE_LABELS,
} from "./config/constants";
import { computeGlobalRanges } from "./utils/dataTransformers";
import ScenarioSelector from "./components/ScenarioSelector";
import YearToggle from "./components/YearToggle";
import DisplayToggle from "./components/DisplayToggle";
import BaselineToggle from "./components/BaselineToggle";
import CapSelector from "./components/CapSelector";
import RevenueByCapChart from "./components/RevenueByCapChart";
import RevenueSummary from "./components/RevenueSummary";
import DecileChart from "./components/DecileChart";
import AffectedShareChart from "./components/AffectedShareChart";
import { formatCap } from "./utils/formatters";

function getInitialState() {
  const params = new URLSearchParams(window.location.search);
  return {
    cap: Number(params.get("cap")) || 2000,
    year: Number(params.get("year")) || 2029,
    employer: params.get("employer") || "spread",
    employee: params.get("employee") || "maintain",
    display: params.get("display") || "relative",
    baseline: params.get("baseline") || "none",
  };
}

function syncQueryParams(state) {
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

export default function App() {
  const { data, loading, error } = useSimulationData();

  const initial = getInitialState();
  const [cap, setCap] = useState(initial.cap);
  const [year, setYear] = useState(initial.year);
  const [employer, setEmployer] = useState(initial.employer);
  const [employee, setEmployee] = useState(initial.employee);
  const [display, setDisplay] = useState(initial.display);
  const [baseline, setBaseline] = useState(initial.baseline);

  const scenario = `${employer}_${employee === "cash" ? "cash" : "maintain"}`;

  const ranges = useMemo(() => {
    if (!data?.results) return null;
    return computeGlobalRanges(data.results, baseline);
  }, [data, baseline]);

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

  const baselineLabel =
    baseline === "none" ? "no cap" : `${BASELINE_LABELS[baseline]}`;
  const decileDomain =
    display === "relative" ? ranges?.pct : ranges?.abs;

  return (
    <>
      <header className="app-header">
        <h1>UK salary sacrifice cap analysis</h1>
        <p>
          Fiscal and distributional impacts of capping salary sacrifice
          arrangements
        </p>
      </header>
      <div className="app-container">
        <div className="controls-bar">
          <BaselineToggle baseline={baseline} onChange={setBaseline} />
          <ScenarioSelector
            employer={employer}
            employee={employee}
            onEmployerChange={setEmployer}
            onEmployeeChange={setEmployee}
          />
          <YearToggle year={year} onChange={setYear} />
          <DisplayToggle display={display} onChange={setDisplay} />
        </div>
        <div className="cap-slider-group">
          <h3>Cap level</h3>
          <CapSelector cap={cap} onChange={setCap} />
        </div>
        <div className="chart-section">
          <h2>Revenue by cap level</h2>
          <p className="chart-subtitle">
            vs {baselineLabel} &middot; {SCENARIO_LABELS[scenario]},{" "}
            {YEAR_LABELS[year]}
          </p>
          <div className="chart-container" style={{ height: 350 }}>
            <RevenueByCapChart
              data={data.results}
              year={year}
              scenario={scenario}
              cap={cap}
              baseline={baseline}
              yDomain={ranges?.revenue}
            />
          </div>
        </div>
        <div className="chart-section">
          <h2>Revenue by scenario at {formatCap(cap)} cap</h2>
          <p className="chart-subtitle">
            vs {baselineLabel} &middot; {YEAR_LABELS[year]}
          </p>
          <RevenueSummary
            data={data.results}
            cap={cap}
            year={year}
            baseline={baseline}
          />
        </div>
        <div className="chart-section">
          <h2>Distributional impact by income decile</h2>
          <p className="chart-subtitle">
            vs {baselineLabel} &middot; {SCENARIO_LABELS[scenario]},{" "}
            {formatCap(cap)} cap, {YEAR_LABELS[year]}
          </p>
          <div className="chart-container" style={{ height: 350 }}>
            <DecileChart
              data={data.results}
              cap={cap}
              year={year}
              scenario={scenario}
              display={display}
              baseline={baseline}
              yDomain={decileDomain}
            />
          </div>
        </div>
        <div className="chart-section">
          <h2>Share of households affected by decile</h2>
          <p className="chart-subtitle">
            {SCENARIO_LABELS[scenario]}, {formatCap(cap)} cap,{" "}
            {YEAR_LABELS[year]}
          </p>
          <div className="chart-container" style={{ height: 350 }}>
            <AffectedShareChart
              data={data.results}
              cap={cap}
              year={year}
              scenario={scenario}
              yDomain={ranges?.share}
            />
          </div>
        </div>
      </div>
      <footer className="app-footer">
        Analysis by{" "}
        <a href="https://policyengine.org">PolicyEngine</a> |{" "}
        <a href="https://github.com/PolicyEngine/uk-salary-sacrifice-analysis">
          Source code
        </a>
      </footer>
    </>
  );
}
