import { CAPS, SCENARIO_KEYS } from "../config/constants";

/**
 * Get data for a specific cap/year/scenario combo.
 */
export function getScenarioData(results, cap, year, scenario) {
  return results?.[String(cap)]?.[String(year)]?.[scenario] ?? null;
}

/**
 * Get revenue across all cap levels for a given year and scenario,
 * optionally adjusted relative to a baseline cap.
 * Returns array of { cap, revenue_bn } sorted by cap.
 */
export function getRevenueAcrossCaps(
  results,
  year,
  scenario,
  caps,
  baselineCap = "none",
) {
  const baselineRevenue =
    baselineCap !== "none"
      ? getScenarioData(results, Number(baselineCap), year, scenario)
          ?.revenue_bn ?? 0
      : 0;

  return caps.map((cap) => {
    const data = getScenarioData(results, cap, year, scenario);
    const raw = data?.revenue_bn ?? null;
    return {
      cap,
      revenue_bn: raw != null ? raw - baselineRevenue : null,
    };
  });
}

/**
 * Get all 4 scenarios' revenue for a given cap and year,
 * optionally adjusted relative to a baseline cap.
 */
export function getAllScenariosRevenue(
  results,
  cap,
  year,
  scenarioKeys,
  baselineCap = "none",
) {
  const out = {};
  for (const key of scenarioKeys) {
    const data = getScenarioData(results, cap, year, key);
    const raw = data?.revenue_bn ?? null;
    if (raw == null) {
      out[key] = null;
      continue;
    }
    if (baselineCap !== "none") {
      const baselineData = getScenarioData(
        results,
        Number(baselineCap),
        year,
        key,
      );
      out[key] = raw - (baselineData?.revenue_bn ?? 0);
    } else {
      out[key] = raw;
    }
  }
  return out;
}

/**
 * Get distributional data for a cap/year/scenario, adjusted for baseline.
 * Returns { deciles, pct_change, abs_change, share_affected } or null.
 */
export function getDistributional(results, cap, year, scenario, baselineCap) {
  const data = getScenarioData(results, cap, year, scenario);
  const dist = data?.distributional;
  if (!dist) return null;

  if (baselineCap === "none") return dist;

  const baseData = getScenarioData(
    results,
    Number(baselineCap),
    year,
    scenario,
  );
  const baseDist = baseData?.distributional;
  if (!baseDist) return dist;

  return {
    deciles: dist.deciles,
    pct_change: dist.pct_change.map(
      (v, i) => v - (baseDist.pct_change[i] ?? 0),
    ),
    abs_change: dist.abs_change.map(
      (v, i) => v - (baseDist.abs_change[i] ?? 0),
    ),
    share_affected: dist.share_affected,
  };
}

/**
 * Compute global y-axis ranges across all caps, scenarios, and years
 * for consistent axes regardless of selection.
 */
export function computeGlobalRanges(results, baselineCap) {
  let minRevenue = Infinity;
  let maxRevenue = -Infinity;
  let minPct = Infinity;
  let maxPct = -Infinity;
  let minAbs = Infinity;
  let maxAbs = -Infinity;
  let maxShare = 0;

  for (const cap of CAPS) {
    for (const year of [2029, 2030]) {
      for (const scenario of SCENARIO_KEYS) {
        // Revenue
        const revData = getRevenueAcrossCaps(
          results,
          year,
          scenario,
          [cap],
          baselineCap,
        );
        const rev = revData[0]?.revenue_bn;
        if (rev != null) {
          minRevenue = Math.min(minRevenue, rev);
          maxRevenue = Math.max(maxRevenue, rev);
        }

        // Distributional
        const dist = getDistributional(
          results,
          cap,
          year,
          scenario,
          baselineCap,
        );
        if (dist) {
          for (const v of dist.pct_change) {
            minPct = Math.min(minPct, v);
            maxPct = Math.max(maxPct, v);
          }
          for (const v of dist.abs_change) {
            minAbs = Math.min(minAbs, v);
            maxAbs = Math.max(maxAbs, v);
          }
          for (const v of dist.share_affected) {
            maxShare = Math.max(maxShare, v * 100);
          }
        }
      }
    }
  }

  // Round to nice axis boundaries
  const niceFloor = (v, step) => Math.floor(v / step) * step;
  const niceCeil = (v, step) => Math.ceil(v / step) * step;

  return {
    revenue: [
      niceFloor(Math.min(0, minRevenue), 1),
      niceCeil(maxRevenue, 1),
    ],
    pct: [niceFloor(minPct, 0.1), niceCeil(Math.max(0, maxPct), 0.1)],
    abs: [niceFloor(minAbs, 100), niceCeil(Math.max(0, maxAbs), 100)],
    share: [0, niceCeil(maxShare, 1)],
  };
}
