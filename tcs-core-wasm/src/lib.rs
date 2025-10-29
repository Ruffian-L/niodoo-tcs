use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmEmbeddingBuffer {
    inner: tcs_core::embeddings::EmbeddingBuffer,
}

#[wasm_bindgen]
impl WasmEmbeddingBuffer {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> WasmEmbeddingBuffer {
        WasmEmbeddingBuffer {
            inner: tcs_core::embeddings::EmbeddingBuffer::new(capacity),
        }
    }

    pub fn push_scalar(&mut self, value: f32) {
        self.inner.push(vec![value]);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }
}
