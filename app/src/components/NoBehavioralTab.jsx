import React, { useState, useEffect } from "react";
import { useNoBehavioralData } from "../hooks/useNoBehavioralData";
import {
  YEAR_LABELS,
  BASELINE_LABELS,
  COLORS,
  FONT_FAMILY,
  CAPS,
} from "../config/constants";
import YearToggle from "./YearToggle";
import DisplayToggle from "./DisplayToggle";
import BaselineToggle from "./BaselineToggle";
import CapSelector from "./CapSelector";
import InfoTooltip from "./InfoTooltip";
import { formatCap, formatBillions, computeNiceAxis } from "../utils/formatters";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
  LabelList,
  Cell,
} from "recharts";

function getInitialState() {
  const params = new URLSearchParams(window.location.search);
  return {
    cap: Number(params.get("cap")) || 5000,
    year: Number(params.get("year")) || 2029,
    display: params.get("display") || "absolute",
    baseline: params.get("baseline") || "2000",
  };
}

function syncQueryParams(state) {
  const params = new URLSearchParams(window.location.search);
  params.set("cap", state.cap);
  params.set("year", state.year);
  params.set("display", state.display);
  params.set("baseline", state.baseline);
  params.set("tab", "no-behavioral");
  const newUrl =
    window.location.pathname + "?" + params.toString() + window.location.hash;
  window.history.replaceState(null, "", newUrl);
}

// Data accessors for new structure: results[baseline][cap][year]
function getDataForBaseline(results, baseline, cap, year) {
  return results?.[baseline]?.[cap]?.[year] ?? null;
}

function getRevenueData(results, baseline, year, caps) {
  return caps.map((cap) => {
    const data = getDataForBaseline(results, baseline, cap, year);
    return {
      cap,
      revenue_bn: data?.revenue_bn ?? null,
    };
  });
}

function getDistributional(results, baseline, cap, year) {
  const data = getDataForBaseline(results, baseline, cap, year);
  return data?.distributional ?? null;
}

function getWinnersLosers(results, baseline, cap, year) {
  const data = getDataForBaseline(results, baseline, cap, year);
  return data?.winners_losers ?? null;
}


function RevenueByCapChartNB({ data, year, cap, baseline }) {
  const revenueData = getRevenueData(data.results, baseline, year, CAPS);
  const chartData = revenueData
    .filter((d) => d.revenue_bn != null)
    .map((d) => ({
      label: formatCap(d.cap),
      revenue_bn: d.revenue_bn,
      cap: d.cap,
      barLabel: `£${parseFloat(d.revenue_bn.toFixed(1))}bn`,
    }));

  const hasNegative = chartData.some((d) => d.revenue_bn < 0);
  const { domain: yDomain, ticks: yTicks } = computeNiceAxis(chartData.map((d) => d.revenue_bn));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 30, right: 20, bottom: 40, left: 15 }}
        barCategoryGap="20%"
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="label"
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <YAxis
          domain={yDomain}
          ticks={yTicks}
          allowDataOverflow
          label={{
            value: "Revenue (£bn)",
            angle: -90,
            position: "insideLeft",
            style: {
              fontFamily: FONT_FAMILY,
              fontSize: 12,
              textAnchor: "middle",
            },
          }}
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <Tooltip
          formatter={(value) => [
            `£${parseFloat(value.toFixed(2))}bn`,
            "Revenue",
          ]}
          contentStyle={{ fontFamily: FONT_FAMILY }}
        />
        {hasNegative && (
          <ReferenceLine y={0} stroke={COLORS.teal800} strokeWidth={1} />
        )}
        <Bar dataKey="revenue_bn">
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.cap === cap ? COLORS.teal700 : COLORS.teal500}
            />
          ))}
          <LabelList
            dataKey="barLabel"
            content={({ x, y, width, height, value, index }) => {
              const isNeg = chartData[index]?.revenue_bn < 0;
              const barTop = Math.min(y, y + height);
              const barBottom = Math.max(y, y + height);
              return (
                <text
                  x={x + width / 2}
                  y={isNeg ? barBottom + 14 : barTop - 6}
                  textAnchor="middle"
                  style={{ fontFamily: FONT_FAMILY, fontSize: 10, fill: COLORS.teal700 }}
                >
                  {value}
                </text>
              );
            }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function DecileChartNB({ data, cap, year, display, baseline }) {
  const dist = getDistributional(data.results, baseline, cap, year);
  if (!dist) return <p>No data available</p>;

  const isRelative = display === "relative";
  const yValues = isRelative ? dist.pct_change : dist.abs_change;

  const formatPct = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const formatCurrency = (v) =>
    `${v >= 0 ? "+" : "-"}£${Math.abs(v).toFixed(0)}`;

  const chartData = dist.deciles.map((d, i) => {
    const value = yValues[i] ?? 0;
    let fill;
    if (value < 0) fill = COLORS.gray600;
    else if (value > 0) fill = COLORS.teal500;
    else fill = COLORS.gray400;

    return {
      decile: String(d),
      value,
      fill,
      label: isRelative ? formatPct(value) : formatCurrency(value),
    };
  });

  const { domain: yDomain, ticks: yTicks } = computeNiceAxis(chartData.map((d) => d.value));

  const formatYTick = (value) => {
    if (isRelative) return `${value}%`;
    return `£${value}`;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 30, right: 20, bottom: 50, left: 15 }}
        barCategoryGap="20%"
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="decile"
          label={{
            value: "Income decile",
            position: "insideBottom",
            offset: -10,
            style: { fontFamily: FONT_FAMILY, fontSize: 12 },
          }}
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <YAxis
          domain={yDomain}
          ticks={yTicks}
          allowDataOverflow
          label={{
            value: "Change in household income",
            angle: -90,
            position: "insideLeft",
            style: {
              fontFamily: FONT_FAMILY,
              fontSize: 12,
              textAnchor: "middle",
            },
          }}
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
          tickFormatter={formatYTick}
        />
        <Tooltip
          formatter={(value) => [
            isRelative ? formatPct(value) : formatCurrency(value),
            "Change",
          ]}
          contentStyle={{ fontFamily: FONT_FAMILY }}
        />
        <ReferenceLine y={0} stroke={COLORS.teal800} strokeWidth={2} />
        <Bar dataKey="value">
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
          <LabelList
            dataKey="label"
            content={({ x, y, width, height, value, index }) => {
              const item = chartData[index];
              const isNeg = (item?.value ?? 0) < 0;
              const labelFill = isNeg ? COLORS.gray600 : COLORS.teal700;
              const barTop = Math.min(y, y + height);
              const barBottom = Math.max(y, y + height);
              return (
                <text
                  x={x + width / 2}
                  y={isNeg ? barBottom + 13 : barTop - 5}
                  textAnchor="middle"
                  style={{ fontFamily: FONT_FAMILY, fontSize: 10, fill: labelFill }}
                >
                  {value}
                </text>
              );
            }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function RevenueSummaryNB({ data, cap, year, baseline }) {
  const result = getDataForBaseline(data.results, baseline, cap, year);
  const revenue = result?.revenue_bn ?? null;

  return (
    <div className="revenue-grid" style={{ gridTemplateColumns: "1fr" }}>
      <div className="revenue-card">
        <div className="label">No behavioural response</div>
        <div className="value">
          {revenue != null ? formatBillions(revenue) : "\u2014"}
        </div>
      </div>
    </div>
  );
}

function WinnersLosersChart({ data, cap, year, baseline }) {
  const wl = getWinnersLosers(data.results, baseline, cap, year);
  if (!wl) return <p>No winners/losers data available</p>;
  const chartData = wl.deciles.map((d, i) => ({
    decile: String(d),
    winners: wl.pct_winners[i],
    losers: -wl.pct_losers[i],
    noChange: wl.pct_no_change[i],
    winnersLabel: `${wl.pct_winners[i].toFixed(1)}%`,
    losersLabel: `${wl.pct_losers[i].toFixed(1)}%`,
  }));

  const { domain: yDomain, ticks: yTicks } = computeNiceAxis([
    ...chartData.map((d) => d.winners),
    ...chartData.map((d) => d.losers),
  ]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 30, right: 20, bottom: 50, left: 15 }}
        barCategoryGap="20%"
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="decile"
          label={{
            value: "Income decile",
            position: "insideBottom",
            offset: -10,
            style: { fontFamily: FONT_FAMILY, fontSize: 12 },
          }}
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <YAxis
          domain={yDomain}
          ticks={yTicks}
          allowDataOverflow
          tickFormatter={(v) => `${Math.abs(v)}%`}
          label={{
            value: "% of people",
            angle: -90,
            position: "insideLeft",
            style: {
              fontFamily: FONT_FAMILY,
              fontSize: 12,
              textAnchor: "middle",
            },
          }}
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <ReferenceLine y={0} stroke={COLORS.teal800} strokeWidth={2} />
        <Tooltip
          formatter={(v, name) => {
            const absVal = Math.abs(v);
            const labels = { winners: "Winners", losers: "Losers" };
            return [`${absVal.toFixed(1)}%`, labels[name]];
          }}
          contentStyle={{ fontFamily: FONT_FAMILY }}
        />
        <Bar
          dataKey="winners"
          fill={COLORS.teal500}
          stackId="stack"
          name="winners"
        >
          <LabelList
            dataKey="winnersLabel"
            content={({ x, y, width, value, index }) => {
              if (chartData[index]?.winners === 0) return null;
              return (
                <text
                  x={x + width / 2}
                  y={y - 5}
                  textAnchor="middle"
                  style={{ fontFamily: FONT_FAMILY, fontSize: 10, fill: COLORS.teal700 }}
                >
                  {value}
                </text>
              );
            }}
          />
        </Bar>
        <Bar
          dataKey="losers"
          fill={COLORS.gray600}
          stackId="stack"
          name="losers"
        >
          <LabelList
            dataKey="losersLabel"
            content={({ x, y, width, height, value, index }) => {
              if (chartData[index]?.losers === 0) return null;
              const barBottom = Math.max(y, y + height);
              return (
                <text
                  x={x + width / 2}
                  y={barBottom + 13}
                  textAnchor="middle"
                  style={{ fontFamily: FONT_FAMILY, fontSize: 10, fill: COLORS.gray600 }}
                >
                  {value}
                </text>
              );
            }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function NoBehavioralTab() {
  const { data, loading, error } = useNoBehavioralData();

  const initial = getInitialState();
  const [cap, setCap] = useState(initial.cap);
  const [year, setYear] = useState(initial.year);
  const [display, setDisplay] = useState(initial.display);
  const [baseline, setBaseline] = useState(initial.baseline);

  useEffect(() => {
    syncQueryParams({ cap, year, display, baseline });
  }, [cap, year, display, baseline]);

  if (loading) {
    return (
      <div className="loading-state">
        <p>Loading no-behavioural simulation data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <p>
          No behavioural response data not yet generated. Run:{" "}
          <code>salary-sacrifice generate-no-behavioral</code>
        </p>
      </div>
    );
  }

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
        would apply. This tool shows the static impact assuming no behavioural
        response from employers or employees, while the behavioural tab models
        different scenarios for how employers and employees might adjust their
        contributions in response to the cap. See the{" "}
        <a href="https://www.policyengine.org/uk/research/uk-salary-sacrifice-cap">
          full report
        </a>{" "}
        for methodology.
      </p>
      <div className="controls-card">
        <div className="controls-card-header">
          <h2>Analysis settings</h2>
          <button
            className="calculate-btn"
            onClick={() =>
              document
                .querySelector(".chart-section")
                ?.scrollIntoView({ behavior: "smooth" })
            }
          >
            View results &darr;
          </button>
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
      </div>
      <div className="chart-section">
        <h2>Revenue by cap level</h2>
        <p className="section-description">
          This chart shows estimated government revenue at each cap level
          assuming no behavioural response. The highlighted bar indicates the
          currently selected cap.
        </p>
        <div className="chart-container" style={{ height: 350 }}>
          <RevenueByCapChartNB
            data={data}
            year={year}
            cap={cap}
            baseline={baseline}
          />
        </div>
      </div>
      <div className="chart-section">
        <h2>Revenue at {formatCap(cap)} cap</h2>
        <p className="section-description">
          This shows the total estimated revenue at the selected cap level,
          assuming no behavioural response from employers or employees.
        </p>
        <RevenueSummaryNB data={data} cap={cap} year={year} baseline={baseline} />
      </div>
      <div className="chart-grid">
        <div className="chart-section">
          <h2>Distributional impact by income decile</h2>
          <p className="section-description">
            This chart shows the average change in household disposable income
            for each income decile at the selected cap level.
          </p>
          <DisplayToggle display={display} onChange={setDisplay} />
          <div className="chart-container">
            <DecileChartNB
              data={data}
              cap={cap}
              year={year}
              display={display}
              baseline={baseline}
            />
          </div>
        </div>
        <div className="chart-section">
          <h2>Winners and losers by income decile</h2>
          <p className="section-description">
            This chart shows the percentage of people in each income decile who
            gain, lose, or are unaffected by the cap.
          </p>
          <div className="chart-container">
            <WinnersLosersChart
              data={data}
              cap={cap}
              year={year}
              baseline={baseline}
            />
          </div>
        </div>
      </div>
    </>
  );
}
