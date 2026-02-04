import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import { CAPS, COLORS, FONT_FAMILY } from "../config/constants";
import { formatCap } from "../utils/formatters";
import { getRevenueAcrossCaps } from "../utils/dataTransformers";

export default function RevenueByCapChart({ data, year, scenario, cap }) {
  const revenueData = getRevenueAcrossCaps(data, year, scenario, CAPS).filter(
    (d) => d.revenue_bn != null,
  );

  const chartData = revenueData.map((d) => ({
    label: formatCap(d.cap),
    revenue_bn: d.revenue_bn,
    cap: d.cap,
  }));

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
          label={{
            value: "Revenue (\u00a3bn)",
            angle: -90,
            position: "insideLeft",
            style: { fontFamily: FONT_FAMILY, fontSize: 12, textAnchor: "middle" },
          }}
          tick={{ fontFamily: FONT_FAMILY, fontSize: 12 }}
        />
        <Tooltip
          formatter={(value) => [`\u00a3${parseFloat(value.toFixed(1))}bn`, "Revenue"]}
          contentStyle={{ fontFamily: FONT_FAMILY }}
        />
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
