import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import CapSelector from "../components/CapSelector";

// CapSelector renders a <select> populated with the configured CAPS.
describe("CapSelector", () => {
  it("renders with the correct formatted value selected", () => {
    render(<CapSelector cap={2000} onChange={() => {}} />);
    expect(screen.getByRole("combobox")).toHaveValue("2000");
  });

  it("renders an option for the formatted current value", () => {
    render(<CapSelector cap={5000} onChange={() => {}} />);
    expect(screen.getByRole("option", { name: "£5k" })).toBeInTheDocument();
  });

  it("includes both the £0 and £6k boundary options", () => {
    render(<CapSelector cap={2000} onChange={() => {}} />);
    expect(screen.getByRole("option", { name: "£0" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "£6k" })).toBeInTheDocument();
  });

  it("calls onChange with numeric value when a different option is chosen", () => {
    const handleChange = vi.fn();
    render(<CapSelector cap={2000} onChange={handleChange} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "4000" } });
    expect(handleChange).toHaveBeenCalledWith(4000);
  });

  it("renders the £0 option even when the current cap is 0", () => {
    render(<CapSelector cap={0} onChange={() => {}} />);
    expect(screen.getByRole("combobox")).toHaveValue("0");
    expect(screen.getByRole("option", { name: "£0" })).toBeInTheDocument();
  });
});
