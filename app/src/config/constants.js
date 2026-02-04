export const COLORS = {
  teal500: "#319795",
  teal300: "#4FD1C5",
  teal700: "#285E61",
  teal800: "#1D4044",
  gray600: "#4B5563",
  gray400: "#9CA3AF",
  gray200: "#E5E7EB",
  gray100: "#F3F4F6",
  gray50: "#F9FAFB",
  white: "#FFFFFF",
};

export const CAPS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
export const REFERENCE_CAP = 2000;

export const SCENARIO_KEYS = [
  "spread_maintain",
  "spread_cash",
  "absorb_maintain",
  "absorb_cash",
];

export const SCENARIO_LABELS = {
  spread_maintain: "Spread cost + Maintain pension",
  spread_cash: "Spread cost + Take cash",
  absorb_maintain: "Absorb cost + Maintain pension",
  absorb_cash: "Absorb cost + Take cash",
};

export const EMPLOYER_LABELS = {
  spread: "Spread cost",
  absorb: "Absorb cost",
};

export const EMPLOYEE_LABELS = {
  maintain: "Maintain pension",
  cash: "Take cash",
};

export const YEAR_LABELS = {
  2029: "2029-30",
  2030: "2030-31",
};

export const BASELINE_LABELS = {
  none: "No cap (pre-budget)",
  "2000": "£2k cap (current law)",
};

export const FONT_FAMILY =
  "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
