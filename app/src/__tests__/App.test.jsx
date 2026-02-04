import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import App from "../App";
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

beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve(mockData),
    }),
  );
});

describe("App", () => {
  it("shows loading state initially", () => {
    render(<App />);
    expect(screen.getByText("Loading simulation data...")).toBeInTheDocument();
  });

  it("renders header after data loads", async () => {
    render(<App />);
    await waitFor(() => {
      expect(
        screen.getByText("UK salary sacrifice cap analysis"),
      ).toBeInTheDocument();
    });
  });

  it("renders scenario controls after data loads", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Spread cost")).toBeInTheDocument();
    });
    expect(screen.getByText("Absorb cost")).toBeInTheDocument();
    expect(screen.getByText("Maintain pension")).toBeInTheDocument();
    expect(screen.getByText("Take cash")).toBeInTheDocument();
  });

  it("renders year toggle after data loads", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("2029-30").length).toBeGreaterThan(0);
    });
    expect(screen.getByText("2030-31")).toBeInTheDocument();
  });

  it("renders display toggle after data loads", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Relative (%)")).toBeInTheDocument();
    });
    expect(screen.getByText("Absolute (\u00a3)")).toBeInTheDocument();
  });

  it("renders chart section headings after data loads", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Revenue by cap level")).toBeInTheDocument();
    });
    expect(
      screen.getByText(/Revenue by scenario at/),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Distributional impact by income decile"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Share of households affected by decile"),
    ).toBeInTheDocument();
  });

  it("renders footer with links", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("PolicyEngine")).toBeInTheDocument();
    });
    expect(screen.getByText("Source code")).toBeInTheDocument();
  });

  it("shows error state when fetch fails", async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: false,
        status: 500,
      }),
    );
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Failed to load data/)).toBeInTheDocument();
    });
  });
});
