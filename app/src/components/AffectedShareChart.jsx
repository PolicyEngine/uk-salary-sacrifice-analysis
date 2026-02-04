import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  LabelList,
} from "recharts";
import { COLORS, FONT_FAMILY } from "../config/constants";
import { getScenarioData } from "../utils/dataTransformers";

export default function AffectedShareChart({ data, cap, year, scenario }) {
  const scenarioData = getScenarioData(data, cap, year, scenario);
  const distributional = scenarioData?.distributional;

  if (!distributional) return null;

  const deciles = distributional.deciles ?? [];
  const shareValues = (distributional.share_affected ?? []).map((v) => v * 100);

  const chartData = deciles.map((d, i) => ({
    decile: String(d),
    share: shareValues[i] ?? 0,
    label: `${(shareValues[i] ?? 0).toFixed(1)}%`,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 30, right: 20, bottom: 60, left: 60 }}
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
            value: "Share of households affected (%)",
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
          formatter={(value) => [`${parseFloat(value.toFixed(1))}%`, "Affected"]}
          contentStyle={{ fontFamily: FONT_FAMILY }}
        />
        <Bar dataKey="share" fill={COLORS.teal500}>
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
