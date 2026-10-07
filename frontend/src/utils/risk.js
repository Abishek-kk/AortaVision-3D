export function getRiskColor(value) {
  if (value < 50) return "#22C55E";
  if (value <= 70) return "#FACC15";
  return "#FF3B5C";
}

export function getRiskLevel(value) {
  if (value < 50) return "low";
  if (value <= 70) return "moderate";
  return "high";
}
