export const DEFAULT_SHAP = [
  { feature: "Age", value: 21 },
  { feature: "Blood Pressure", value: 12 },
  { feature: "LDL", value: 7 },
  { feature: "Echo RWMA", value: 5 },
  { feature: "Temperature", value: 4 },
  { feature: "Heart Rate", value: 3 },
  { feature: "Airway disease", value: 3 },
  { feature: "Paroxysmal AF", value: 2 },
  { feature: "Smoking history", value: 2 },
];

export const DEFAULT_RESULT = {
  cad: { present: true, probability: 0.72 },
  vessels: { LAD: 85, LCX: 45, RCA: 60 },
  shap: DEFAULT_SHAP,
  summary: "Age and Elevated Cholesterol are top risk contributors",
};
