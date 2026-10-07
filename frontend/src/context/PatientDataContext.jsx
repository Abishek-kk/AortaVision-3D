import { createContext, useContext, useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "heartcare-ai-patient-data";
const PatientDataContext = createContext(null);

function readStoredData() {
  try {
    const stored = window.sessionStorage.getItem(STORAGE_KEY);
    if (!stored) return { uploadedFiles: [], predictionResult: null };

    const parsed = JSON.parse(stored);
    return {
      uploadedFiles: Array.isArray(parsed.uploadedFiles)
        ? parsed.uploadedFiles
        : [],
      predictionResult: parsed.predictionResult?.predictions
        ? parsed.predictionResult
        : null,
    };
  } catch (error) {
    console.warn("Unable to restore patient data from session storage.", error);
    return { uploadedFiles: [], predictionResult: null };
  }
}

export function PatientDataProvider({ children }) {
  const [data, setData] = useState(readStoredData);

  useEffect(() => {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      console.warn("Unable to persist patient data to session storage.", error);
    }
  }, [data]);

  const value = useMemo(
    () => ({
      ...data,
      setUploadedFiles: (uploadedFiles) =>
        setData((current) => ({ ...current, uploadedFiles })),
      setPredictionResult: (predictionResult) =>
        setData((current) => ({ ...current, predictionResult })),
      clearPatientData: () =>
        setData({ uploadedFiles: [], predictionResult: null }),
    }),
    [data],
  );

  return (
    <PatientDataContext.Provider value={value}>
      {children}
    </PatientDataContext.Provider>
  );
}

export function usePatientData() {
  const context = useContext(PatientDataContext);
  if (!context) {
    throw new Error("usePatientData must be used within PatientDataProvider.");
  }
  return context;
}
