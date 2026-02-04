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
} from "recharts";
import { CAPS, COLORS, FONT_FAMILY } from "../config/constants";
import { formatCap } from "../utils/formatters";
import { getRevenueAcrossCaps } from "../utils/dataTransformers";

export default function RevenueByCapChart({
  data,
  year,
  scenario,
  cap,
  baseline,
  yDomain,
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
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
