import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LabelList,
  ReferenceLine,
} from "recharts";
import { COLORS, FONT_FAMILY } from "../config/constants";
import { computeNiceDomain } from "../utils/formatters";
import { getScenarioData, getDistributional } from "../utils/dataTransformers";

export default function AffectedShareChart({
  data,
  cap,
  year,
  scenario,
  baseline,
}) {
  const scenarioData = getScenarioData(data, cap, year, scenario);
  const rawDist = scenarioData?.distributional;
  const adjDist = getDistributional(data, cap, year, scenario, baseline);

  if (!rawDist || !adjDist) return null;

  const deciles = rawDist.deciles ?? [];
  const shareValues = (rawDist.share_affected ?? []).map((v) => v * 100);
  const adjPctChange = adjDist.pct_change ?? [];

  const chartData = deciles.map((d, i) => {
    const share = shareValues[i] ?? 0;
    const change = adjPctChange[i] ?? 0;
    const winners = change > 0 ? share : 0;
    const losers = change < 0 ? -share : 0;
    return {
      decile: String(d),
      winners,
      losers,
      winnersLabel: `${winners.toFixed(1)}%`,
      losersLabel: `${share.toFixed(1)}%`,
    };
  });

  const yDomain = computeNiceDomain([
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
