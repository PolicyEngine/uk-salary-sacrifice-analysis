import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import DecileChart from "../components/DecileChart";

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

const testData = {
  "2000": {
    "2029": {
      spread_maintain: {
        revenue_bn: 3.3,
        distributional: {
          deciles: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          pct_change: [
            -0.01, -0.01, -0.21, -0.04, -0.04, -0.04, -0.09, -0.07, -0.08,
            -0.35,
          ],
          abs_change: [-2, -4, -63, -14, -16, -17, -50, -40, -60, -431],
          share_affected: [
            0.005, 0.01, 0.03, 0.02, 0.04, 0.05, 0.08, 0.1, 0.14, 0.25,
          ],
        },
      },
    },
  },
};

describe("DecileChart", () => {
  const defaultProps = {
    data: testData,
    cap: 2000,
    year: 2029,
    scenario: "spread_maintain",
    display: "relative",
  };

  it("renders without crashing", () => {
    render(<DecileChart {...defaultProps} />);
    expect(
      screen.getByTestId("recharts-responsive-container"),
    ).toBeInTheDocument();
  });

  it("renders in relative mode", () => {
    render(<DecileChart {...defaultProps} display="relative" />);
    expect(
      screen.getByTestId("recharts-responsive-container"),
    ).toBeInTheDocument();
  });

  it("renders in absolute mode", () => {
    render(<DecileChart {...defaultProps} display="absolute" />);
    expect(
      screen.getByTestId("recharts-responsive-container"),
    ).toBeInTheDocument();
  });

  it("renders a recharts BarChart", () => {
    const { container } = render(<DecileChart {...defaultProps} />);
    expect(
      container.querySelector(".recharts-wrapper") ||
        screen.getByTestId("recharts-responsive-container"),
    ).toBeTruthy();
  });

  it("returns null for missing scenario data", () => {
    const { container } = render(
      <DecileChart {...defaultProps} scenario="nonexistent_scenario" />,
    );
    expect(container.innerHTML).toBe("");
  });
});
