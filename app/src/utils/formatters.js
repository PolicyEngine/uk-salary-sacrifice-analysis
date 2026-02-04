/**
 * Format a currency value as -£400 (not £-400).
 * Rounds to nearest integer.
 */
export function formatCurrency(value) {
  const rounded = Math.round(value);
  if (rounded < 0) {
    return `-£${Math.abs(rounded).toLocaleString("en-GB")}`;
  }
  return `£${rounded.toLocaleString("en-GB")}`;
}

/**
 * Format a percentage value (input in percentage points, e.g. -0.35).
 * Rounds to 1 decimal place.
 */
export function formatPct(value) {
  const rounded = parseFloat(value.toFixed(1));
  return `${rounded}%`;
}

/**
 * Format billions as £X.Xbn.
 * Rounds to 1 decimal place.
 */
export function formatBillions(value) {
  const rounded = parseFloat(value.toFixed(1));
  if (rounded < 0) {
    return `-£${Math.abs(rounded)}bn`;
  }
  return `£${rounded}bn`;
}

/**
 * Format a cap value as £Xk.
 */
export function formatCap(value) {
  if (value === 0) return "£0";
  return `£${(value / 1000).toFixed(0)}k`;
}
