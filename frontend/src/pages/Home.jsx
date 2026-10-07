import Papa from "papaparse";
import { ArrowRight, FileText, LoaderCircle, X } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import GlowButton from "../components/GlowButton.jsx";
import GlowCard from "../components/GlowCard.jsx";
import HeartGlow from "../components/HeartGlow.jsx";
import { usePatientData } from "../context/PatientDataContext.jsx";
import { SAMPLE_PATIENT } from "../data/samplePatient.js";
import { predictRisk } from "../services/predict.js";

const ACCEPTED_EXTENSIONS = new Set(["csv", "json"]);
const MAX_FILE_SIZE = 10 * 1024 * 1024;

function fileExtension(name) {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

function formatFileSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function FileTypeIcon({ extension, className = "h-6 w-6" }) {
  const iconColor = {
    csv: "text-green-600",
    json: "text-navy",
  }[extension] ?? "text-slate-500";

  return <FileText aria-hidden="true" className={`${className} ${iconColor}`} />;
}

function parseUpload(file, extension) {
  if (extension === "json") {
    return file.text().then((contents) => {
      let parsedData;
      try {
        parsedData = JSON.parse(contents);
      } catch (error) {
        throw new Error(`Invalid JSON: ${error.message}`);
      }
      return { name: file.name, size: file.size, type: file.type, parsedData };
    });
  }

  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        const fatalError = result.errors.find(
          (error) => error.type === "Delimiter" || error.type === "Quotes",
        );
        if (fatalError) {
          reject(new Error(`Unable to parse CSV: ${fatalError.message}`));
          return;
        }
        resolve({
          name: file.name,
          size: file.size,
          type: file.type,
          parsedData: result.data,
        });
      },
      error: (error) => reject(new Error(`Unable to parse CSV: ${error.message}`)),
    });
  });
}

function EdgeEcg({ side }) {
  return (
    <svg
      aria-hidden="true"
      className={`pointer-events-none absolute top-0 h-full w-[18%] opacity-[0.13] blur-[1px] ${
        side === "left" ? "left-0" : "right-0 scale-x-[-1]"
      }`}
      preserveAspectRatio="none"
      viewBox="0 0 240 700"
    >
      <path
        d="M-30 340h45l17-3 14-25 17 55 22-88 19 64 14-11 70 1 25-2"
        fill="none"
        stroke="#38BDF8"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="3"
      />
      <path
        d="M-20 346h47l15-2 16-20 15 45 20-74 19 56 17-9 80 1"
        fill="none"
        stroke="#38BDF8"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1"
      />
    </svg>
  );
}

export default function Home() {
  const navigate = useNavigate();
  const { uploadedFiles, setUploadedFiles, setPredictionResult } =
    usePatientData();
  const inputRef = useRef(null);
  const filesRef = useRef(uploadedFiles);
  const [errors, setErrors] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  function updateFiles(nextFiles) {
    filesRef.current = nextFiles;
    setUploadedFiles(nextFiles);
  }

  function addError(message) {
    setErrors((current) => [...current, message]);
  }

  async function processFiles(fileList) {
    setErrors([]);
    const selectedFiles = Array.from(fileList);
    if (selectedFiles.length > 1) {
      addError("Select one CSV or JSON file containing a single patient record.");
      return;
    }
    for (const file of selectedFiles) {
      const extension = fileExtension(file.name);
      if (!ACCEPTED_EXTENSIONS.has(extension)) {
        addError(`${file.name}: unsupported file type. Choose CSV or JSON.`);
        continue;
      }
      if (file.size > MAX_FILE_SIZE) {
        addError(`${file.name}: exceeds the 10 MB file size limit.`);
        continue;
      }
      if (
        filesRef.current.some(
          (existing) => existing.name === file.name && existing.size === file.size,
        )
      ) {
        addError(`${file.name}: this file has already been added.`);
        continue;
      }

      try {
        const uploadedFile = await parseUpload(file, extension);
        updateFiles([uploadedFile]);
      } catch (error) {
        addError(`${file.name}: ${error.message}`);
      }
    }
  }

  function removeFile(index) {
    updateFiles(filesRef.current.filter((_, fileIndex) => fileIndex !== index));
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    void processFiles(event.dataTransfer.files);
  }

  function handleDropZoneKeyDown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      inputRef.current?.click();
    }
  }

  function useSampleData() {
    const sampleJson = JSON.stringify(SAMPLE_PATIENT, null, 2);
    const sampleFile = {
      name: "sample-patient.json",
      size: new Blob([sampleJson]).size,
      type: "application/json",
      parsedData: SAMPLE_PATIENT,
      isSample: true,
    };
    updateFiles([sampleFile]);
    setErrors([]);
  }

  async function handleAnalyze() {
    if (filesRef.current.length === 0 || isAnalyzing) return;
    setIsAnalyzing(true);
    try {
      const submittedFiles = filesRef.current;
      setUploadedFiles(submittedFiles);
      const result = await predictRisk({ uploadedFiles: submittedFiles });
      setPredictionResult(result);
      navigate("/analyze");
    } catch (error) {
      addError(`Unable to analyze these files: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <section className="relative isolate flex min-h-full flex-1 flex-col items-center overflow-hidden px-4 pb-6 pt-4 sm:px-8 sm:pt-6">
      <EdgeEcg side="left" />
      <EdgeEcg side="right" />
      <div className="absolute right-[3%] top-3 z-0 w-36 opacity-90 sm:right-[4%] sm:top-5 sm:w-44 md:right-[3%] md:w-48 lg:right-[2%] lg:top-4 lg:w-48">
        <HeartGlow />
      </div>

      <div className="relative z-10 flex w-full max-w-5xl flex-col items-center">
        <div className="mb-3 flex min-h-20 w-full flex-col items-center justify-end text-center sm:mb-4 sm:min-h-24">
          <h1 className="max-w-4xl text-3xl font-extrabold leading-tight tracking-wide sm:text-4xl md:text-5xl lg:max-w-[74%] lg:text-4xl">
            PATIENT DATA INTAKE &amp; UPLOAD{" "}
            <span className="text-cyan">(LANDING PAGE)</span>
          </h1>
        </div>

        <GlowCard
          className="!bg-slate-50 w-full max-w-[1100px] text-navy"
          title="UPLOAD PATIENT REPORTS"
        >
          <div className="bg-slate-50 p-3 sm:p-4 md:p-5">
            <div
              aria-label="Upload a CSV or JSON patient file"
              className={`flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-4 text-center transition-colors duration-200 sm:min-h-52 ${
                isDragging
                  ? "border-cyan bg-lightBlue"
                  : "border-navy/70 bg-white hover:border-cyan hover:bg-lightBlue/50"
              }`}
              onClick={() => inputRef.current?.click()}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) {
                  setIsDragging(false);
                }
              }}
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
              onKeyDown={handleDropZoneKeyDown}
              role="button"
              tabIndex={0}
            >
              <div className="mb-2 flex items-center justify-center gap-7 sm:gap-10">
                {["csv", "json"].map((extension) => (
                  <div
                    key={extension}
                    className="flex flex-col items-center gap-1 text-xs font-extrabold uppercase tracking-wider text-navy"
                  >
                    <FileTypeIcon extension={extension} className="h-8 w-8 sm:h-9 sm:w-9" />
                    {extension}
                  </div>
                ))}
              </div>
              <button
                className="rounded-full border border-navy bg-navy px-6 py-2 text-sm font-bold uppercase tracking-wider text-white transition hover:bg-midNavy"
                onClick={(event) => {
                  event.stopPropagation();
                  inputRef.current?.click();
                }}
                type="button"
              >
                Choose Files
              </button>
              <p className="mt-3 text-base font-semibold text-navy sm:text-lg">
                Drag &amp; drop files or click to upload
              </p>
              <p className="mt-1 text-sm text-slate-600">
                CSV or JSON · one patient record · up to 10 MB
              </p>
              <input
                ref={inputRef}
                accept=".csv,.json,text/csv,application/json"
                aria-label="Choose patient report files"
                className="sr-only"
                onClick={(event) => event.stopPropagation()}
                onChange={(event) => {
                  void processFiles(event.target.files);
                  event.target.value = "";
                }}
                type="file"
              />
            </div>

            {uploadedFiles.length > 0 && (
              <div aria-live="polite" className="mt-5">
                <p className="mb-2 font-bold text-green-700">
                  {uploadedFiles.length}{" "}
                  {uploadedFiles.length === 1 ? "file" : "files"} uploaded
                </p>
                <ul className="space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <li
                      key={`${file.name}-${file.size}-${index}`}
                      className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2"
                    >
                      <FileTypeIcon extension={fileExtension(file.name)} />
                      <span className="min-w-0 flex-1 truncate text-left font-semibold">
                        {file.name}
                        <span className="ml-2 text-sm font-normal text-slate-500">
                          {formatFileSize(file.size)}
                        </span>
                      </span>
                      <button
                        aria-label={`Remove ${file.name}`}
                        className="rounded-full p-2 text-slate-500 transition hover:bg-red-50 hover:text-red-600"
                        onClick={() => removeFile(index)}
                        type="button"
                      >
                        <X aria-hidden="true" className="h-4 w-4" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {errors.length > 0 && (
              <ul aria-live="assertive" className="mt-4 space-y-1 text-left text-sm font-semibold text-red-700">
                {errors.map((error, index) => (
                  <li key={`${error}-${index}`}>{error}</li>
                ))}
              </ul>
            )}
          </div>
        </GlowCard>

        <div className="mt-4 flex flex-col items-center gap-2 text-center">
          <span
            className="group relative"
            title={uploadedFiles.length === 0 ? "Upload at least one file" : undefined}
          >
            <GlowButton
              className="gap-2 border-2 px-7 py-2.5 text-base uppercase tracking-wider sm:px-9"
              disabled={uploadedFiles.length === 0 || isAnalyzing}
              onClick={handleAnalyze}
            >
              {isAnalyzing ? (
                <>
                  <LoaderCircle aria-hidden="true" className="h-5 w-5 animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  Analyze Cardiac Risk <ArrowRight aria-hidden="true" className="h-5 w-5" />
                </>
              )}
            </GlowButton>
            {uploadedFiles.length === 0 && (
              <span
                className="pointer-events-none absolute bottom-full left-1/2 mb-2 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-deepNavy px-3 py-1.5 text-xs text-white group-hover:block group-focus-within:block"
                role="tooltip"
              >
                Upload at least one file
              </span>
            )}
          </span>
          <button
            className="text-sm font-semibold text-cyan underline-offset-4 transition hover:text-white hover:underline"
            onClick={useSampleData}
            type="button"
          >
            Use sample patient data
          </button>
        </div>
      </div>
    </section>
  );
}
