import { writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const outputPath = resolve("frontend/public/heart_mesh.gltf");

function pushVec(target, x, y, z) {
  target.push(x, y, z);
}

function makeEllipsoid(name, center, scale, rows = 20, cols = 32) {
  const positions = [];
  const normals = [];
  const indices = [];
  for (let r = 0; r <= rows; r += 1) {
    const v = r / rows;
    const theta = v * Math.PI;
    for (let c = 0; c <= cols; c += 1) {
      const u = c / cols;
      const phi = u * Math.PI * 2;
      const nx = Math.sin(theta) * Math.cos(phi);
      const ny = Math.cos(theta);
      const nz = Math.sin(theta) * Math.sin(phi);
      pushVec(positions, center[0] + nx * scale[0], center[1] + ny * scale[1], center[2] + nz * scale[2]);
      pushVec(normals, nx, ny, nz);
    }
  }
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const a = r * (cols + 1) + c;
      const b = a + cols + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return { name, positions, normals, indices };
}

function makeCone(name, center, radius, height, segments = 48) {
  const positions = [];
  const normals = [];
  const indices = [];
  const apex = [center[0], center[1] - height * 0.5, center[2]];
  const baseY = center[1] + height * 0.5;
  pushVec(positions, apex[0], apex[1], apex[2]);
  pushVec(normals, 0, -1, 0);
  for (let i = 0; i <= segments; i += 1) {
    const angle = (i / segments) * Math.PI * 2;
    const x = center[0] + Math.cos(angle) * radius;
    const z = center[2] + Math.sin(angle) * radius;
    pushVec(positions, x, baseY, z);
    pushVec(normals, Math.cos(angle), 0.35, Math.sin(angle));
  }
  for (let i = 1; i <= segments; i += 1) {
    indices.push(0, i, i + 1);
  }
  return { name, positions, normals, indices };
}

function makeCylinder(name, center, radiusTop, radiusBottom, height, segments = 48) {
  const positions = [];
  const normals = [];
  const indices = [];
  for (let yIndex = 0; yIndex <= 1; yIndex += 1) {
    const y = center[1] + (yIndex - 0.5) * height;
    const radius = yIndex === 0 ? radiusBottom : radiusTop;
    for (let i = 0; i <= segments; i += 1) {
      const angle = (i / segments) * Math.PI * 2;
      pushVec(positions, center[0] + Math.cos(angle) * radius, y, center[2] + Math.sin(angle) * radius);
      pushVec(normals, Math.cos(angle), 0, Math.sin(angle));
    }
  }
  for (let i = 0; i < segments; i += 1) {
    const a = i;
    const b = i + segments + 1;
    indices.push(a, b, a + 1, b, b + 1, a + 1);
  }
  return { name, positions, normals, indices };
}

function floatBuffer(values) {
  const buffer = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => buffer.writeFloatLE(value, index * 4));
  return buffer;
}

function indexBuffer(values) {
  const buffer = Buffer.alloc(values.length * 2);
  values.forEach((value, index) => buffer.writeUInt16LE(value, index * 2));
  return buffer;
}

function bounds(values) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < values.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], values[i + axis]);
      max[axis] = Math.max(max[axis], values[i + axis]);
    }
  }
  return { min, max };
}

const meshes = [
  makeEllipsoid("left_ventricular_surface", [-0.32, 0.34, 0], [0.82, 1.06, 0.78]),
  makeEllipsoid("right_ventricular_surface", [0.36, 0.28, -0.02], [0.76, 1.0, 0.72]),
  makeCone("cardiac_apex_surface", [0, -0.56, 0.04], 0.92, 1.5),
  makeCylinder("ascending_aorta_surface", [0.08, 1.08, -0.1], 0.36, 0.26, 1.05),
];

const buffers = [];
const bufferViews = [];
const accessors = [];
const gltfMeshes = [];

function addBufferView(buffer, target) {
  const byteOffset = buffers.reduce((sum, current) => sum + current.length, 0);
  buffers.push(buffer);
  bufferViews.push({ buffer: 0, byteOffset, byteLength: buffer.length, target });
  return bufferViews.length - 1;
}

function addAccessor(bufferView, componentType, count, type, min, max) {
  accessors.push({ bufferView, componentType, count, type, min, max });
  return accessors.length - 1;
}

for (const mesh of meshes) {
  const positionView = addBufferView(floatBuffer(mesh.positions), 34962);
  const normalView = addBufferView(floatBuffer(mesh.normals), 34962);
  const indexView = addBufferView(indexBuffer(mesh.indices), 34963);
  const meshBounds = bounds(mesh.positions);
  const positionAccessor = addAccessor(positionView, 5126, mesh.positions.length / 3, "VEC3", meshBounds.min, meshBounds.max);
  const normalAccessor = addAccessor(normalView, 5126, mesh.normals.length / 3, "VEC3");
  const indexAccessor = addAccessor(indexView, 5123, mesh.indices.length, "SCALAR", [0], [Math.max(...mesh.indices)]);
  gltfMeshes.push({
    name: mesh.name,
    primitives: [
      {
        attributes: { POSITION: positionAccessor, NORMAL: normalAccessor },
        indices: indexAccessor,
        material: mesh.name.includes("aorta") ? 1 : 0,
      },
    ],
  });
}

const combined = Buffer.concat(buffers);
const gltf = {
  asset: { version: "2.0", generator: "AortaVision procedural heart surface generator" },
  scene: 0,
  scenes: [{ name: "AortaVision heart surface scene", nodes: [0] }],
  nodes: [
    {
      name: "heart_surface_root_no_coronary_vessels",
      children: [1, 2, 3, 4],
    },
    ...gltfMeshes.map((mesh, index) => ({ name: mesh.name, mesh: index })),
  ],
  meshes: gltfMeshes,
  materials: [
    {
      name: "myocardium_surface_material",
      pbrMetallicRoughness: { baseColorFactor: [0.55, 0.13, 0.18, 1], metallicFactor: 0, roughnessFactor: 0.58 },
    },
    {
      name: "great_vessel_surface_material",
      pbrMetallicRoughness: { baseColorFactor: [0.72, 0.28, 0.3, 1], metallicFactor: 0, roughnessFactor: 0.48 },
    },
  ],
  buffers: [{ uri: `data:application/octet-stream;base64,${combined.toString("base64")}`, byteLength: combined.length }],
  bufferViews,
  accessors,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(gltf, null, 2)}\n`, "utf8");
console.log(`Wrote ${outputPath}`);
