export const GLOSSARY_CATEGORIES = [
  { id: "diseases", label: "Diseases", color: "rose" },
  { id: "arteries", label: "Arteries", color: "orange" },
  { id: "valves", label: "Valves", color: "violet" },
  { id: "measurements", label: "Measurements", color: "emerald" },
];

export const GLOSSARY_TERMS = [
  {
    id: "cad",
    category: "diseases",
    title: "Coronary Artery Disease (CAD)",
    short: "Narrowing of the arteries that supply blood to the heart.",
    definition:
      "Coronary artery disease develops when fatty deposits called plaque build up in the coronary arteries and reduce blood flow to the heart muscle.",
    context:
      "CAD can cause chest discomfort or breathlessness, but some people have no symptoms. Risk assessment combines symptoms, history, tests, and imaging; an AI estimate is not a diagnosis.",
    relatedConditions: ["Atherosclerosis", "Angina", "Heart attack", "Stenosis"],
  },
  {
    id: "stenosis",
    category: "diseases",
    title: "Stenosis",
    short: "An abnormal narrowing of a blood vessel or heart valve.",
    definition:
      "Stenosis means that an artery, valve, or other passage has become narrower than usual. In a coronary artery, plaque is a common cause of narrowing.",
    context:
      "A narrowing may limit blood flow, especially during activity. Its importance depends on where it is, how severe it is, symptoms, and clinical test results.",
    relatedConditions: ["Coronary artery disease", "Atherosclerosis", "Angina"],
  },
  {
    id: "angina",
    category: "diseases",
    title: "Angina",
    short: "Chest discomfort caused by reduced blood flow to the heart.",
    definition:
      "Angina is chest pressure, tightness, or discomfort that occurs when the heart muscle does not receive enough oxygen-rich blood.",
    context:
      "It can happen with exertion or stress and may spread to the arm, jaw, back, or shoulder. New, severe, or persistent chest pain needs urgent medical attention.",
    relatedConditions: ["Coronary artery disease", "Stenosis", "Myocardial infarction"],
  },
  {
    id: "myocardial-infarction",
    category: "diseases",
    title: "Myocardial Infarction",
    short: "A heart attack caused by injury to heart muscle from blocked blood flow.",
    definition:
      "A myocardial infarction (heart attack) occurs when blood flow to part of the heart muscle is suddenly blocked or severely reduced, injuring that muscle.",
    context:
      "Possible symptoms include chest pressure, shortness of breath, sweating, nausea, or unusual weakness. Call emergency services for suspected heart attack.",
    relatedConditions: ["Coronary artery disease", "Angina", "ECG/EKG"],
  },
  {
    id: "lad",
    category: "arteries",
    title: "LAD (Left Anterior Descending)",
    short: "Supplies blood to much of the front of the heart.",
    definition:
      "The left anterior descending artery is a major coronary artery that travels along the front of the heart and supplies blood to the front wall and much of the septum.",
    context:
      "The LAD is one of the main branches of the left coronary artery. Clinicians interpret any suspected narrowing together with symptoms and appropriate tests.",
    relatedConditions: ["Coronary artery disease", "Stenosis", "Angina"],
  },
  {
    id: "lcx",
    category: "arteries",
    title: "LCX (Left Circumflex)",
    short: "Supplies blood to the side and, in many people, the back of the heart.",
    definition:
      "The left circumflex artery curves around the left side of the heart. Its branches commonly supply the side and sometimes the back of the heart.",
    context:
      "The exact area supplied varies between people. A clinician considers that variation when interpreting imaging or a possible LCX narrowing.",
    relatedConditions: ["Coronary artery disease", "Stenosis", "Angina"],
  },
  {
    id: "rca",
    category: "arteries",
    title: "RCA (Right Coronary Artery)",
    short: "Usually supplies the right side and part of the lower heart.",
    definition:
      "The right coronary artery runs along the right side of the heart and commonly supplies the right ventricle and part of the lower heart.",
    context:
      "Its branching pattern varies. The RCA may also supply structures involved in the heart's electrical rhythm.",
    relatedConditions: ["Coronary artery disease", "Stenosis", "Heart rhythm disorders"],
  },
  {
    id: "aorta",
    category: "arteries",
    title: "Aorta",
    short: "The large artery that carries blood from the heart to the body.",
    definition:
      "The aorta is the body's main artery. It receives oxygen-rich blood from the left ventricle and distributes it through branches to the body.",
    context:
      "The aorta is a large systemic artery, not a coronary artery. Its wall and valve can be assessed with clinical examination and imaging.",
    relatedConditions: ["Aortic valve disease", "Aneurysm", "Hypertension"],
  },
  {
    id: "pulmonary-artery",
    category: "arteries",
    title: "Pulmonary Artery",
    short: "Carries oxygen-poor blood from the heart to the lungs.",
    definition:
      "The pulmonary artery carries blood from the right ventricle to the lungs, where it picks up oxygen.",
    context:
      "Unlike most arteries, the pulmonary artery carries oxygen-poor blood. It is part of the pulmonary circulation.",
    relatedConditions: ["Pulmonary hypertension", "Right ventricle"],
  },
  {
    id: "vena-cava",
    category: "arteries",
    title: "Vena Cava",
    short: "Returns oxygen-poor blood from the body to the heart.",
    definition:
      "The superior and inferior vena cava are large veins that return blood from the upper and lower body to the right atrium.",
    context:
      "Veins carry blood toward the heart. The vena cava connects the body's circulation with the right side of the heart.",
    relatedConditions: ["Right atrium", "Right ventricle"],
  },
  {
    id: "left-atrium",
    category: "measurements",
    title: "Left Atrium",
    short: "Receives oxygen-rich blood from the lungs before it enters the left ventricle.",
    definition:
      "The left atrium is the upper-left chamber of the heart. It receives oxygen-rich blood from the pulmonary veins and passes it through the mitral valve.",
    context:
      "Its size and function may be assessed with heart imaging. The left atrium is part of the route that sends oxygenated blood to the body.",
    relatedConditions: ["Mitral valve", "Atrial fibrillation", "Left ventricle"],
  },
  {
    id: "right-atrium",
    category: "measurements",
    title: "Right Atrium",
    short: "Receives oxygen-poor blood returning from the body.",
    definition:
      "The right atrium is the upper-right chamber of the heart. It receives blood from the vena cava and passes it to the right ventricle.",
    context:
      "The right atrium is part of the route that moves blood to the lungs for oxygenation.",
    relatedConditions: ["Vena cava", "Right ventricle", "Heart rhythm disorders"],
  },
  {
    id: "mitral-valve",
    category: "valves",
    title: "Mitral Valve",
    short: "Controls blood flow from the left atrium to the left ventricle.",
    definition:
      "The mitral valve sits between the left atrium and left ventricle. It opens to let blood pass into the ventricle and closes to help prevent backflow.",
    context:
      "The valve normally opens and closes with each heartbeat. Narrowing or leakage can affect how efficiently the heart moves blood.",
    relatedConditions: ["Mitral stenosis", "Mitral regurgitation", "Heart failure"],
  },
  {
    id: "aortic-valve",
    category: "valves",
    title: "Aortic Valve",
    short: "Controls blood leaving the left ventricle and entering the aorta.",
    definition:
      "The aortic valve opens when the left ventricle contracts and closes to help stop blood flowing back from the aorta.",
    context:
      "Aortic stenosis limits forward flow, while aortic regurgitation allows backward leakage. A clinician evaluates suspected valve disease with examination and imaging.",
    relatedConditions: ["Aortic stenosis", "Aortic regurgitation", "Aorta"],
  },
  {
    id: "left-ventricle",
    category: "measurements",
    title: "Left Ventricle",
    short: "The heart chamber that pumps oxygen-rich blood to the body.",
    definition:
      "The left ventricle is the heart's main pumping chamber for systemic circulation. It sends blood through the aortic valve into the aorta.",
    context:
      "Its size, movement, and pumping function can be assessed with imaging such as echocardiography.",
    relatedConditions: ["Ejection fraction", "RWMA", "Aortic valve"],
  },
  {
    id: "right-ventricle",
    category: "measurements",
    title: "Right Ventricle",
    short: "Pumps oxygen-poor blood from the heart to the lungs.",
    definition:
      "The right ventricle sends blood through the pulmonary valve and pulmonary artery to the lungs.",
    context:
      "The right ventricle works with the lungs and pulmonary circulation. Its function may be assessed by echocardiography or other imaging.",
    relatedConditions: ["Pulmonary artery", "Pulmonary hypertension", "Ejection fraction"],
  },
  {
    id: "ejection-fraction",
    category: "measurements",
    title: "Ejection Fraction (EF)",
    short: "The percentage of blood the left ventricle pumps out with each beat.",
    definition:
      "Ejection fraction estimates how much blood in the left ventricle is pumped out during each contraction. It is usually measured with an echocardiogram.",
    context:
      "A commonly cited normal left-ventricular EF is about 50–70%, though ranges vary by method and clinical context. EF alone does not describe every aspect of heart function.",
    relatedConditions: ["Left ventricle", "Heart failure", "RWMA"],
  },
  {
    id: "ecg",
    category: "measurements",
    title: "ECG / EKG",
    short: "A tracing that records the heart's electrical activity.",
    definition:
      "An electrocardiogram (ECG or EKG) uses skin electrodes to record the timing and pattern of electrical signals that trigger heartbeats.",
    context:
      "ST elevation, ST depression, T-wave inversion, or voltage patterns suggesting left ventricular hypertrophy (LVH) can be associated with heart conditions, but none is diagnostic on its own.",
    relatedConditions: ["Myocardial infarction", "Arrhythmia", "LVH"],
  },
  {
    id: "ldl",
    category: "measurements",
    title: "LDL Cholesterol",
    short: "A type of cholesterol carried in the blood and linked to plaque buildup.",
    definition:
      "Low-density lipoprotein (LDL) carries cholesterol through the bloodstream. Higher LDL levels can contribute to plaque buildup in artery walls.",
    context:
      "The LDL level considered appropriate depends on a person's overall cardiovascular risk and medical history. A healthcare professional can interpret a lipid test.",
    relatedConditions: ["Atherosclerosis", "Coronary artery disease", "Stenosis"],
  },
  {
    id: "blood-pressure",
    category: "measurements",
    title: "Blood Pressure",
    short: "The force of blood against artery walls, recorded as two numbers.",
    definition:
      "Blood pressure is recorded as systolic pressure over diastolic pressure. Systolic is measured as the heart beats; diastolic is measured between beats.",
    context:
      "Interpretation depends on repeated measurements, health history, and clinical guidance. Persistently high blood pressure can strain the heart and blood vessels.",
    relatedConditions: ["Hypertension", "Coronary artery disease", "Stroke"],
  },
  {
    id: "rwma",
    category: "measurements",
    title: "RWMA",
    short: "A regional wall-motion abnormality seen during heart imaging.",
    definition:
      "Regional wall-motion abnormality describes an area of the heart muscle that moves differently from expected during a heartbeat, often assessed by echocardiography.",
    context:
      "An RWMA can have several causes and must be interpreted alongside symptoms, image quality, and other test results by a qualified clinician.",
    relatedConditions: ["Myocardial infarction", "Ischemia", "Left ventricle"],
  },
];
