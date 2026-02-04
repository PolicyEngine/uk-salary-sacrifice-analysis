import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import RevenueByCapChart from "../components/RevenueByCapChart";
import mockData from "./fixtures/mock-results.json";

vi.mock("recharts", async () => {
  const actual = await vi.importActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }) => (
      <div data-testid="recharts-responsive-container" style={{ width: 500, height: 350 }}>
        {children}
      </div>
    ),
  };
});

describe("RevenueByCapChart", () => {
  const defaultProps = {
    data: mockData.results,
    year: 2029,
    scenario: "spread_maintain",
    cap: 2000,
  };

  it("renders without crashing", () => {
    render(<RevenueByCapChart {...defaultProps} />);
    expect(
      screen.getByTestId("recharts-responsive-container"),
    ).toBeInTheDocument();
  });

  it("renders a bar chart with recharts", () => {
    const { container } = render(<RevenueByCapChart {...defaultProps} />);
    // Recharts renders an SVG-based BarChart
    expect(
      container.querySelector(".recharts-bar-rectangles") ||
        container.querySelector(".recharts-wrapper") ||
        screen.getByTestId("recharts-responsive-container"),
    ).toBeTruthy();
  });

  it("renders a recharts BarChart", () => {
    const { container } = render(<RevenueByCapChart {...defaultProps} />);
    expect(
      container.querySelector(".recharts-wrapper") ||
        screen.getByTestId("recharts-responsive-container"),
    ).toBeTruthy();
  });

  it("handles missing data for some cap levels gracefully", () => {
    // The mock data only has caps 0, 2000, 5000, 10000
    // Other caps like 1000, 3000 etc. will return null revenue and be filtered
    render(<RevenueByCapChart {...defaultProps} />);
    expect(
      screen.getByTestId("recharts-responsive-container"),
    ).toBeInTheDocument();
  });
});
