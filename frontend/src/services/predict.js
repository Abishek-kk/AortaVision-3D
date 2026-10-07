import { DEFAULT_RESULT, DEFAULT_SHAP } from "../data/demoPrediction.js";

function normalizeKey(key) {
  return String(key).toLowerCase().replace(/[^a-z0-9]/g, "");
}

function findNumericValue(value, aliases) {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findNumericValue(item, aliases);
      if (found !== undefined) return found;
    }
    return undefined;
  }

  if (!value || typeof value !== "object") return undefined;

  const aliasSet = new Set(aliases.map(normalizeKey));
  for (const [key, item] of Object.entries(value)) {
    if (aliasSet.has(normalizeKey(key))) {
      const numericValue = Number(item);
      if (Number.isFinite(numericValue)) return numericValue;
    }
  }

  for (const item of Object.values(value)) {
    const found = findNumericValue(item, aliases);
    if (found !== undefined) return found;
  }
  return undefined;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export async function predictRisk(patientData) {
  await new Promise((resolve) => window.setTimeout(resolve, 1500));

  const source = patientData?.uploadedFiles ?? patientData ?? {};
  const age = findNumericValue(source, ["age", "patient_age"]);
  const bloodPressure = findNumericValue(source, [
    "bp",
    "blood pressure",
    "systolic blood pressure",
    "systolic_bp",
  ]);
  const ldl = findNumericValue(source, ["ldl", "ldl cholesterol"]);

  const ageContribution =
    age === undefined ? 21 : clamp(Math.round((age - 30) * 0.55), 0, 35);
  const bloodPressureContribution =
    bloodPressure === undefined
      ? 12
      : clamp(Math.round((bloodPressure - 90) * 0.18), 0, 25);
  const ldlContribution =
    ldl === undefined ? 7 : clamp(Math.round((ldl - 70) * 0.08), 0, 20);

  const probability = clamp(
    DEFAULT_RESULT.cad.probability +
      (ageContribution + bloodPressureContribution + ldlContribution - 40) /
        100,
    0.05,
    0.95,
  );
  const vesselAdjustment = clamp(
    Math.round((probability - DEFAULT_RESULT.cad.probability) * 45),
    -20,
    20,
  );

  // Replace with real XGBoost/SHAP API call here.
  return {
    cad: { present: probability >= 0.5, probability },
    vessels: {
      LAD: clamp(DEFAULT_RESULT.vessels.LAD + vesselAdjustment, 0, 100),
      LCX: clamp(DEFAULT_RESULT.vessels.LCX + vesselAdjustment, 0, 100),
      RCA: clamp(DEFAULT_RESULT.vessels.RCA + vesselAdjustment, 0, 100),
    },
    shap: DEFAULT_SHAP.map((item) => {
      if (item.feature === "Age") return { ...item, value: ageContribution };
      if (item.feature === "Blood Pressure") {
        return { ...item, value: bloodPressureContribution };
      }
      if (item.feature === "LDL") return { ...item, value: ldlContribution };
      return { ...item };
    }),
    summary: "Age and Elevated Cholesterol are top risk contributors",
  };
}
