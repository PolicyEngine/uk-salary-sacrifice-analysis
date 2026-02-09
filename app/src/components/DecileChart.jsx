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
  CartesianGrid,
  LabelList,
} from "recharts";
import { COLORS, FONT_FAMILY } from "../config/constants";
import { formatPct, formatCurrency, computeNiceAxis } from "../utils/formatters";
import { getDistributional } from "../utils/dataTransformers";

export default function DecileChart({
  data,
  cap,
  year,
  scenario,
  display,
  baseline,
}) {
  const distributional = getDistributional(data, cap, year, scenario, baseline);

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

  const { domain: yDomain, ticks: yTicks } = computeNiceAxis(chartData.map((d) => d.value));

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
