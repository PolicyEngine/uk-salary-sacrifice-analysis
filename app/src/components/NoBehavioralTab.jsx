import React, { useState, useEffect, useMemo } from "react";
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
import { formatCap, formatBillions } from "../utils/formatters";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
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

function computeGlobalRanges(results, baseline) {
  let minRev = 0,
    maxRev = 0;
  let minPct = 0,
    maxPct = 0;
  let minAbs = 0,
    maxAbs = 0;

  for (const year of [2029, 2030]) {
    const revenueData = getRevenueData(results, baseline, year, CAPS);
    revenueData.forEach(({ revenue_bn }) => {
      if (revenue_bn != null) {
        minRev = Math.min(minRev, revenue_bn);
        maxRev = Math.max(maxRev, revenue_bn);
      }
    });

    for (const cap of CAPS) {
      const dist = getDistributional(results, baseline, cap, year);
      if (dist) {
        dist.pct_change.forEach((v) => {
          minPct = Math.min(minPct, v);
          maxPct = Math.max(maxPct, v);
        });
        dist.abs_change.forEach((v) => {
          minAbs = Math.min(minAbs, v);
          maxAbs = Math.max(maxAbs, v);
        });
      }
    }
  }

  const niceFloor = (v, step) => Math.floor(v / step) * step;
  const niceCeil = (v, step) => Math.ceil(v / step) * step;

  return {
    revenue: [niceFloor(Math.min(0, minRev), 1), niceCeil(maxRev, 1)],
    pct: [niceFloor(minPct, 0.1), niceCeil(Math.max(0, maxPct), 0.1)],
    abs: [niceFloor(minAbs, 100), niceCeil(Math.max(0, maxAbs), 100)],
  };
}

function RevenueByCapChartNB({ data, year, cap, baseline, yDomain }) {
  const revenueData = getRevenueData(data.results, baseline, year, CAPS);
  const chartData = revenueData
    .filter((d) => d.revenue_bn != null)
    .map((d) => ({
      label: formatCap(d.cap),
      revenue_bn: d.revenue_bn,
      cap: d.cap,
    }));

  const hasNegative = chartData.some((d) => d.revenue_bn < 0);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 10, right: 20, bottom: 40, left: 60 }}
        barCategoryGap="20%"
      >
        <XAxis
          dataKey="label"
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <YAxis
          domain={yDomain}
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
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function DecileChartNB({ data, cap, year, display, baseline, yDomain }) {
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

  const formatYTick = (value) => {
    if (isRelative) return `${value}%`;
    return `£${value}`;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 20, right: 30, bottom: 60, left: 70 }}
        barCategoryGap="20%"
      >
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
        <div className="label">No behavioral response</div>
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
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 20, right: 30, bottom: 60, left: 70 }}
        barCategoryGap="20%"
      >
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
          domain={[-20, 30]}
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
        />
        <Bar
          dataKey="losers"
          fill={COLORS.gray600}
          stackId="stack"
          name="losers"
        />
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

  const ranges = useMemo(() => {
    if (!data?.results) return null;
    return computeGlobalRanges(data.results, baseline);
  }, [data, baseline]);

  useEffect(() => {
    syncQueryParams({ cap, year, display, baseline });
  }, [cap, year, display, baseline]);

  if (loading) {
    return (
      <div className="loading-state">
        <p>Loading no-behavioral simulation data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <p>
          No behavioral response data not yet generated. Run:{" "}
          <code>salary-sacrifice generate-no-behavioral</code>
        </p>
      </div>
    );
  }

  const baselineLabel =
    baseline === "none" ? "no cap" : `${BASELINE_LABELS[baseline]}`;
  const decileDomain = display === "relative" ? ranges?.pct : ranges?.abs;

  return (
    <>
      <div className="controls-bar">
        <BaselineToggle baseline={baseline} onChange={setBaseline} />
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
          vs {baselineLabel} &middot; {YEAR_LABELS[year]}
        </p>
        <div className="chart-container" style={{ height: 350 }}>
          <RevenueByCapChartNB
            data={data}
            year={year}
            cap={cap}
            baseline={baseline}
            yDomain={ranges?.revenue}
          />
        </div>
      </div>
      <div className="chart-section">
        <h2>Revenue at {formatCap(cap)} cap</h2>
        <p className="chart-subtitle">
          vs {baselineLabel} &middot; {YEAR_LABELS[year]}
        </p>
        <RevenueSummaryNB data={data} cap={cap} year={year} baseline={baseline} />
      </div>
      <div className="chart-section">
        <h2>Distributional impact by income decile</h2>
        <p className="chart-subtitle">
          vs {baselineLabel} &middot; {formatCap(cap)} cap, {YEAR_LABELS[year]}
        </p>
        <div className="chart-container" style={{ height: 350 }}>
          <DecileChartNB
            data={data}
            cap={cap}
            year={year}
            display={display}
            baseline={baseline}
            yDomain={decileDomain}
          />
        </div>
      </div>
      <div className="chart-section">
        <h2>Winners and losers by income decile</h2>
        <p className="chart-subtitle">
          {formatCap(cap)} cap, {YEAR_LABELS[year]}
        </p>
        <div className="chart-container" style={{ height: 350 }}>
          <WinnersLosersChart data={data} cap={cap} year={year} baseline={baseline} />
        </div>
      </div>
    </>
  );
}
