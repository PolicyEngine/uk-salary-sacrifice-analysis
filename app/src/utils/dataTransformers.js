/**
 * Get data for a specific cap/year/scenario combo.
 */
export function getScenarioData(results, cap, year, scenario) {
  return results?.[String(cap)]?.[String(year)]?.[scenario] ?? null;
}

/**
 * Get revenue across all cap levels for a given year and scenario.
 * Returns array of { cap, revenue_bn } sorted by cap.
 */
export function getRevenueAcrossCaps(results, year, scenario, caps) {
  return caps.map((cap) => {
    const data = getScenarioData(results, cap, year, scenario);
    return {
      cap,
      revenue_bn: data?.revenue_bn ?? null,
    };
  });
}

/**
 * Get all 4 scenarios' revenue for a given cap and year.
 * Returns object keyed by scenario name.
 */
export function getAllScenariosRevenue(results, cap, year, scenarioKeys) {
  const out = {};
  for (const key of scenarioKeys) {
    const data = getScenarioData(results, cap, year, key);
    out[key] = data?.revenue_bn ?? null;
  }
  return out;
}
