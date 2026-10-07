import { copyFile, readFile, stat, unlink } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NodeIO, PropertyType } from "@gltf-transform/core";
import { KHRDracoMeshCompression } from "@gltf-transform/extensions";
import { dedup, draco, flatten, join, prune, simplify, weld } from "@gltf-transform/functions";
import draco3d from "draco3d";
import { MeshoptSimplifier } from "meshoptimizer";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelPath = path.join(projectRoot, "public", "models", "heart.glb");
const tempPath = path.join(projectRoot, "public", "models", "heart.optimized.tmp.glb");
const protectedNames = new Set([
  "LAD",
  "LAD_D1",
  "LAD_D2",
  "LAD_D3",
  "LAD_Apical",
  "LCX",
  "LCX_OM1",
  "LCX_OM2",
  "LCX_PL",
  "LCX_Atrial",
  "RCA",
  "RCA_AM1",
  "RCA_AM2",
  "RCA_PDA",
  "RCA_RV",
]);
const TARGET_TRIANGLES = 40_000;
const TARGET_BYTES = 1_000_000;
const OPTIMIZATION_MARKER = "heartcare-model-optimization-v3";

const io = new NodeIO()
  .registerExtensions([KHRDracoMeshCompression])
  .registerDependencies({
    "draco3d.decoder": await draco3d.createDecoderModule(),
    "draco3d.encoder": await draco3d.createEncoderModule(),
  });

await MeshoptSimplifier.ready;

function collectMetrics(document, fileSize) {
  const root = document.getRoot();
  const meshes = root.listMeshes();
  const nodes = root.listNodes();
  const triangles = meshes.reduce(
    (total, mesh) =>
      total +
      mesh.listPrimitives().reduce((meshTotal, primitive) => {
        const indices = primitive.getIndices();
        return meshTotal + (indices ? indices.getCount() / 3 : 0);
      }, 0),
    0,
  );
  const drawCalls = nodes.reduce((total, node) => {
    const mesh = node.getMesh();
    return total + (mesh ? mesh.listPrimitives().length : 0);
  }, 0);

  return {
    triangles,
    drawCalls,
    meshes: meshes.length,
    materials: root.listMaterials().length,
    fileSize,
  };
}

function printMetrics(label, metrics) {
  console.log(
    `${label}: ${metrics.triangles.toLocaleString()} triangles | ` +
      `${metrics.drawCalls} draw calls | ${metrics.meshes} meshes | ` +
      `${metrics.materials} materials | ${(metrics.fileSize / 1024).toFixed(1)} KB`,
  );
}

function validateProtectedMeshes(document) {
  const counts = new Map();
  for (const mesh of document.getRoot().listMeshes()) {
    if (protectedNames.has(mesh.getName())) {
      counts.set(mesh.getName(), (counts.get(mesh.getName()) ?? 0) + 1);
    }
  }

  const missing = [...protectedNames].filter((name) => counts.get(name) !== 1);
  if (missing.length > 0) {
    throw new Error(
      `Required interactive meshes must each exist exactly once: ${missing.join(", ")}`,
    );
  }
}

function validateHeartBody(document) {
  const heartBodies = document
    .getRoot()
    .listMeshes()
    .filter((mesh) => mesh.getName() === "HeartBody");
  if (heartBodies.length !== 1) {
    throw new Error("The optimized model must contain exactly one mesh named HeartBody.");
  }

  const triangles = heartBodies[0].listPrimitives().reduce(
    (total, primitive) => total + (primitive.getIndices()?.getCount() ?? 0) / 3,
    0,
  );
  if (triangles < 20_000 || triangles > 25_000) {
    throw new Error(`HeartBody has ${triangles} triangles; target range is 20,000–25,000.`);
  }
}

const sourceBytes = await readFile(modelPath);
const document = await io.read(modelPath);
const before = collectMetrics(document, sourceBytes.byteLength);
printMetrics("Before", before);

if (document.getRoot().getExtras().optimization === OPTIMIZATION_MARKER) {
  printMetrics("After", before);
  console.log("Model is already optimized; leaving it unchanged.");
  process.exit(0);
}

validateProtectedMeshes(document);

for (const node of document.getRoot().listNodes()) {
  const mesh = node.getMesh();
  if (mesh && mesh.getName() !== "HeartBody" && !protectedNames.has(mesh.getName())) {
    node.setName("");
    mesh.setName("");
  }
}
for (const material of document.getRoot().listMaterials()) {
  material.setName("");
}

await document.transform(
  weld(),
  simplify({
    simplifier: MeshoptSimplifier,
    ratio: 0.3,
    error: 0.01,
  }),
  dedup({ propertyTypes: [PropertyType.MATERIAL] }),
  flatten(),
  join({ keepNamed: true }),
  prune(),
  draco({ method: "edgebreaker", encodeSpeed: 5, decodeSpeed: 5 }),
);

document.getRoot().setExtras({
  ...document.getRoot().getExtras(),
  optimization: OPTIMIZATION_MARKER,
});

try {
  await io.write(tempPath, document);
  const optimizedBytes = await readFile(tempPath);
  const jsonLength = optimizedBytes.readUInt32LE(12);
  const gltf = JSON.parse(optimizedBytes.toString("utf8", 20, 20 + jsonLength));
  if (!gltf.extensionsUsed?.includes("KHR_draco_mesh_compression")) {
    throw new Error("The optimized GLB does not contain Draco-compressed mesh data.");
  }

  const optimizedDocument = await io.read(tempPath);
  const after = collectMetrics(optimizedDocument, (await stat(tempPath)).size);
  validateProtectedMeshes(optimizedDocument);
  validateHeartBody(optimizedDocument);
  printMetrics("After", after);

  if (after.triangles > TARGET_TRIANGLES) {
    throw new Error(
      `Optimized model has ${after.triangles} triangles; target is below ${TARGET_TRIANGLES}.`,
    );
  }
  if (after.fileSize >= TARGET_BYTES) {
    throw new Error(
      `Optimized model is ${after.fileSize} bytes; target is below ${TARGET_BYTES} bytes.`,
    );
  }
  if (after.drawCalls >= before.drawCalls) {
    throw new Error(
      `Draw calls did not decrease (${before.drawCalls} before, ${after.drawCalls} after).`,
    );
  }

  await copyFile(tempPath, modelPath);
  console.log(`Optimized model written to ${path.relative(projectRoot, modelPath)}.`);
} finally {
  await unlink(tempPath).catch((error) => {
    if (error.code !== "ENOENT") throw error;
  });
}
