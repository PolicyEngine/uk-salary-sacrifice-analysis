import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import CapSelector from "../components/CapSelector";

describe("CapSelector", () => {
  it("renders with the correct formatted value displayed", () => {
    render(<CapSelector cap={2000} onChange={() => {}} />);
    expect(screen.getByText("\u00a32k")).toBeInTheDocument();
  });

  it("renders slider with correct attributes", () => {
    render(<CapSelector cap={5000} onChange={() => {}} />);
    const slider = screen.getByRole("slider");
    expect(slider).toBeInTheDocument();
    expect(slider).toHaveValue("5000");
    expect(slider).toHaveAttribute("min", "0");
    expect(slider).toHaveAttribute("max", "10000");
    expect(slider).toHaveAttribute("step", "1000");
  });

  it("displays min and max labels", () => {
    render(<CapSelector cap={2000} onChange={() => {}} />);
    expect(screen.getByText("\u00a30")).toBeInTheDocument();
    expect(screen.getByText("\u00a310k")).toBeInTheDocument();
  });

  it("calls onChange with numeric value when slider moves", () => {
    const handleChange = vi.fn();
    render(<CapSelector cap={2000} onChange={handleChange} />);
    const slider = screen.getByRole("slider");
    fireEvent.change(slider, { target: { value: "4000" } });
    expect(handleChange).toHaveBeenCalledWith(4000);
  });

  it("displays zero cap correctly", () => {
    render(<CapSelector cap={0} onChange={() => {}} />);
    // formatCap(0) returns "£0", and it also appears as the min label
    const zeroElements = screen.getAllByText("\u00a30");
    expect(zeroElements.length).toBeGreaterThanOrEqual(2);
  });
});
