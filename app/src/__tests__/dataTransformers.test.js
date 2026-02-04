import { describe, it, expect } from "vitest";
import {
  getScenarioData,
  getRevenueAcrossCaps,
  getAllScenariosRevenue,
} from "../utils/dataTransformers.js";
import fixture from "./fixtures/mock-results.json";

const results = fixture.results;

describe("getScenarioData", () => {
  it("returns correct data for a valid cap/year/scenario combo", () => {
    const data = getScenarioData(results, 2000, 2029, "spread_maintain");
    expect(data).not.toBeNull();
    expect(data.revenue_bn).toBe(3.3);
    expect(data.affected_workers).toBe(1830770);
    expect(data.total_excess_bn).toBe(18.5);
  });

  it("returns full distributional data for a valid combo", () => {
    const data = getScenarioData(results, 2000, 2029, "spread_maintain");
    expect(data.distributional).toBeDefined();
    expect(data.distributional.deciles).toHaveLength(10);
    expect(data.distributional.pct_change).toHaveLength(10);
    expect(data.distributional.abs_change).toHaveLength(10);
  });

  it("returns correct data for cap=0, year=2029, absorb_cash", () => {
    const data = getScenarioData(results, 0, 2029, "absorb_cash");
    expect(data).not.toBeNull();
    expect(data.revenue_bn).toBe(11.5);
  });

  it("returns correct data for cap=10000, year=2030, spread_cash", () => {
    const data = getScenarioData(results, 10000, 2030, "spread_cash");
    expect(data).not.toBeNull();
    expect(data.revenue_bn).toBe(1.2);
    expect(data.affected_workers).toBe(420000);
  });

  it("returns null for a missing cap", () => {
    const data = getScenarioData(results, 99999, 2029, "spread_maintain");
    expect(data).toBeNull();
  });

  it("returns null for a missing year", () => {
    const data = getScenarioData(results, 2000, 2025, "spread_maintain");
    expect(data).toBeNull();
  });

  it("returns null for a missing scenario", () => {
    const data = getScenarioData(results, 2000, 2029, "nonexistent_scenario");
    expect(data).toBeNull();
  });

  it("returns null when results is null", () => {
    const data = getScenarioData(null, 2000, 2029, "spread_maintain");
    expect(data).toBeNull();
  });

  it("returns null when results is undefined", () => {
    const data = getScenarioData(undefined, 2000, 2029, "spread_maintain");
    expect(data).toBeNull();
  });
});

describe("getRevenueAcrossCaps", () => {
  const caps = [0, 2000, 5000, 10000];

  it("returns an array with revenue for each cap", () => {
    const result = getRevenueAcrossCaps(
      results,
      2029,
      "spread_maintain",
      caps,
    );
    expect(result).toHaveLength(4);
    expect(result[0]).toEqual({ cap: 0, revenue_bn: 5.2 });
    expect(result[1]).toEqual({ cap: 2000, revenue_bn: 3.3 });
    expect(result[2]).toEqual({ cap: 5000, revenue_bn: 1.8 });
    expect(result[3]).toEqual({ cap: 10000, revenue_bn: 0.5 });
  });

  it("returns correct revenue for a different scenario and year", () => {
    const result = getRevenueAcrossCaps(results, 2030, "absorb_cash", caps);
    expect(result).toHaveLength(4);
    expect(result[0]).toEqual({ cap: 0, revenue_bn: 12.0 });
    expect(result[1]).toEqual({ cap: 2000, revenue_bn: 8.0 });
    expect(result[2]).toEqual({ cap: 5000, revenue_bn: 4.4 });
    expect(result[3]).toEqual({ cap: 10000, revenue_bn: 1.3 });
  });

  it("returns null revenue for caps that do not exist in data", () => {
    const capsWithMissing = [0, 7777, 10000];
    const result = getRevenueAcrossCaps(
      results,
      2029,
      "spread_maintain",
      capsWithMissing,
    );
    expect(result).toHaveLength(3);
    expect(result[0].revenue_bn).toBe(5.2);
    expect(result[1]).toEqual({ cap: 7777, revenue_bn: null });
    expect(result[2].revenue_bn).toBe(0.5);
  });

  it("returns all null revenue when results is null", () => {
    const result = getRevenueAcrossCaps(
      null,
      2029,
      "spread_maintain",
      [0, 2000],
    );
    expect(result).toEqual([
      { cap: 0, revenue_bn: null },
      { cap: 2000, revenue_bn: null },
    ]);
  });

  it("returns empty array for empty caps list", () => {
    const result = getRevenueAcrossCaps(results, 2029, "spread_maintain", []);
    expect(result).toEqual([]);
  });
});

describe("getAllScenariosRevenue", () => {
  const scenarioKeys = [
    "spread_maintain",
    "spread_cash",
    "absorb_maintain",
    "absorb_cash",
  ];

  it("returns object with all 4 scenarios' revenue", () => {
    const result = getAllScenariosRevenue(results, 2000, 2029, scenarioKeys);
    expect(result).toEqual({
      spread_maintain: 3.3,
      spread_cash: 7.2,
      absorb_maintain: 3.7,
      absorb_cash: 7.6,
    });
  });

  it("returns correct revenue for cap=0, year=2030", () => {
    const result = getAllScenariosRevenue(results, 0, 2030, scenarioKeys);
    expect(result).toEqual({
      spread_maintain: 5.5,
      spread_cash: 11.5,
      absorb_maintain: 6.1,
      absorb_cash: 12.0,
    });
  });

  it("returns null for all scenarios when cap is missing", () => {
    const result = getAllScenariosRevenue(results, 99999, 2029, scenarioKeys);
    expect(result).toEqual({
      spread_maintain: null,
      spread_cash: null,
      absorb_maintain: null,
      absorb_cash: null,
    });
  });

  it("returns null for all scenarios when year is missing", () => {
    const result = getAllScenariosRevenue(results, 2000, 2025, scenarioKeys);
    expect(result).toEqual({
      spread_maintain: null,
      spread_cash: null,
      absorb_maintain: null,
      absorb_cash: null,
    });
  });

  it("returns subset when given fewer scenario keys", () => {
    const result = getAllScenariosRevenue(results, 5000, 2029, [
      "spread_maintain",
      "absorb_cash",
    ]);
    expect(result).toEqual({
      spread_maintain: 1.8,
      absorb_cash: 4.2,
    });
  });

  it("returns null values for nonexistent scenario keys", () => {
    const result = getAllScenariosRevenue(results, 2000, 2029, [
      "spread_maintain",
      "fake_scenario",
    ]);
    expect(result).toEqual({
      spread_maintain: 3.3,
      fake_scenario: null,
    });
  });
});
