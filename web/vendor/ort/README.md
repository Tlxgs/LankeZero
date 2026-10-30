# ONNX Runtime Web

Vendored files from `onnxruntime-web@1.31.1` (MIT License):

- `ort.webgpu.bundle.min.mjs`
- `ort-wasm-simd-threaded.asyncify.mjs`
- `ort-wasm-simd-threaded.asyncify.wasm`

Source: https://www.npmjs.com/package/onnxruntime-web
License: https://github.com/microsoft/onnxruntime/blob/main/LICENSE

These files are served as static assets by GitHub Pages; the page uses WebGPU when
available and falls back to the WASM execution provider. The asyncify build is the
loader used by the 1.31 WebGPU bundle.
