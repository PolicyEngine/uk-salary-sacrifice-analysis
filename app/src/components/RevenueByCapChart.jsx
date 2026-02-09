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
import { CAPS, COLORS, FONT_FAMILY } from "../config/constants";
import { formatCap, computeNiceAxis } from "../utils/formatters";
import { getRevenueAcrossCaps } from "../utils/dataTransformers";

export default function RevenueByCapChart({
  data,
  year,
  scenario,
  cap,
  baseline,
}) {
  const revenueData = getRevenueAcrossCaps(
    data,
    year,
    scenario,
    CAPS,
    baseline,
  ).filter((d) => d.revenue_bn != null);

  const chartData = revenueData.map((d) => ({
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
            `£${parseFloat(value.toFixed(1))}bn`,
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
