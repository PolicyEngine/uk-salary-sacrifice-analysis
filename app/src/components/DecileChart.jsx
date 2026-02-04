import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
  LabelList,
} from "recharts";
import { COLORS, FONT_FAMILY } from "../config/constants";
import { formatPct, formatCurrency } from "../utils/formatters";
import { getScenarioData } from "../utils/dataTransformers";

export default function DecileChart({ data, cap, year, scenario, display }) {
  const scenarioData = getScenarioData(data, cap, year, scenario);
  const distributional = scenarioData?.distributional;

  if (!distributional) return null;

  const deciles = distributional.deciles ?? [];
  const isRelative = display === "relative";
  const yValues = isRelative
    ? distributional.pct_change ?? []
    : distributional.abs_change ?? [];

  const chartData = deciles.map((d, i) => {
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
    if (isRelative) {
      return `${value}%`;
    }
    return formatCurrency(value);
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
          label={{
            value: "Change in total income",
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
            position="top"
            style={{ fontFamily: FONT_FAMILY, fontSize: 11, fill: "#333" }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
