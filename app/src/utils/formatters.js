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

/**
 * Compute a nice Y-axis domain from an array of values.
 * Always includes 0, adds padding, and rounds to clean tick values.
 */
export function computeNiceDomain(values, padding = 0.2) {
  const filtered = values.filter((v) => v != null && isFinite(v));
  if (filtered.length === 0) return [0, 1];

  let min = Math.min(0, ...filtered);
  let max = Math.max(0, ...filtered);
  const range = max - min || 1;

  min -= range * padding;
  max += range * padding;

  // Pick a clean step size
  const rawStep = range / 4;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const nice = [1, 2, 2.5, 5, 10];
  const step = nice.find((n) => n * magnitude >= rawStep) * magnitude;

  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}
