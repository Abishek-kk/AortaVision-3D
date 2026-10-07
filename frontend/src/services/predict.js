const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000")
  .trim()
  .replace(/\/+$/, "");

const TARGETS = ["CAD", "LAD", "LCX", "RCA"];

function patientFeatures(patientData) {
  if (patientData?.features && typeof patientData.features === "object") {
    return patientData.features;
  }

  const uploadedFiles = patientData?.uploadedFiles;
  if (Array.isArray(uploadedFiles)) {
    if (uploadedFiles.length !== 1) {
      throw new Error("Upload exactly one patient JSON or CSV file per analysis.");
    }

    let data = uploadedFiles[0]?.parsedData;
    if (Array.isArray(data)) {
      if (data.length !== 1) {
        throw new Error("The CSV must contain exactly one patient row.");
      }
      [data] = data;
    }
    if (data?.features && typeof data.features === "object") {
      data = data.features;
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error(
        "The uploaded file must contain a JSON object or a CSV row of patient features.",
      );
    }
    return data;
  }

  if (patientData && typeof patientData === "object") return patientData;
  throw new Error("Patient features were not found in the submitted data.");
}

async function responseError(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    return `Backend returned HTTP ${response.status}.`;
  }
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    return detail.message || JSON.stringify(detail);
  }
  return payload?.message || `Backend returned HTTP ${response.status}.`;
}

export async function predictRisk(patientData) {
  const features = patientFeatures(patientData);
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
  } catch (error) {
    throw new Error(
      `Cannot reach the prediction backend at ${API_BASE_URL}. Start the backend and check VITE_API_URL. (${error.message})`,
    );
  }

  if (!response.ok) {
    throw new Error(await responseError(response));
  }

  const payload = await response.json();
  if (payload?.success !== true || !payload.predictions) {
    throw new Error("The backend returned an invalid prediction response.");
  }

  const missingTargets = TARGETS.filter(
    (target) => !payload.predictions[target],
  );
  if (missingTargets.length > 0) {
    throw new Error(
      `The backend response is missing predictions for: ${missingTargets.join(", ")}.`,
    );
  }

  const cadPrediction = payload.predictions.CAD;
  const shap = (payload.explanations?.CAD ?? []).map((item) => ({
    feature: item.feature,
    value: Number(item.contribution),
    direction: item.direction,
  }));
  const topFactors = shap.slice(0, 2).map((item) => item.feature);

  return {
    cad: {
      present: Number(cadPrediction.prediction) === 1,
      probability: Number(cadPrediction.probability),
      riskLevel: cadPrediction.risk_level,
      threshold: Number(cadPrediction.threshold),
    },
    vessels: Object.fromEntries(
      ["LAD", "LCX", "RCA"].map((target) => [
        target,
        Number(payload.predictions[target].probability) * 100,
      ]),
    ),
    shap,
    summary: topFactors.length
      ? `Top CAD risk contributors: ${topFactors.join(" and ")}.`
      : "The backend did not return feature-contribution explanations.",
    predictions: payload.predictions,
    explanations: payload.explanations ?? {},
    disclaimer: payload.disclaimer ?? "",
  };
}
