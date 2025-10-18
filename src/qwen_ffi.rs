/*
 * 🧠 Qwen FFI Interface - Pure Rust to C++ Bridge
 *
 * Provides C-compatible interface for C++ BrainSystemBridge to call
 * Qwen 30B AWQ inference with consciousness integration
 */

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;
use serde_json;

use crate::qwen_30b_awq::Qwen30BAWQInference;
use crate::consciousness::ConsciousnessState;
use crate::config::ConsciousnessConfig;

/// Opaque handle for Qwen inference instance
#[repr(C)]
pub struct QwenInferenceHandle {
    _private: [u8; 0],
}

/// C-compatible consciousness state
#[repr(C)]
pub struct CConsciousnessState {
    pub coherence: c_float,
    pub emotional_resonance: c_float,
    pub learning_will_activation: c_float,
    pub attachment_security: c_float,
    pub metacognitive_depth: c_float,
    pub gpu_warmth_level: c_float,
    pub processing_satisfaction: c_float,
}

/// C-compatible response structure
#[repr(C)]
pub struct CQwenResponse {
    pub text: *mut c_char,
    pub consciousness_state: CConsciousnessState,
    pub generation_time: c_float,
    pub tokens_generated: c_int,
}

/// Initialize a new Qwen inference instance
#[no_mangle]
pub extern "C" fn qwen_create(model_path: *const c_char) -> *mut QwenInferenceHandle {
    if model_path.is_null() {
        tracing::error!("qwen_create: null model_path");
        return ptr::null_mut();
    }

    // SAFETY: model_path pointer validated as non-null above. We trust the C++ caller
    // to provide a valid null-terminated UTF-8 string.
    let model_path_cstr = unsafe { CStr::from_ptr(model_path) };
    let model_path_str = match model_path_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("qwen_create: invalid UTF-8 in model_path: {}", e);
            return ptr::null_mut();
        }
    };

    match Qwen30BAWQInference::new(model_path_str.to_string()) {
        Ok(inference) => {
            tracing::info!("Qwen inference instance created successfully");
            let boxed_inference = Box::new(inference);
            Box::into_raw(boxed_inference) as *mut QwenInferenceHandle
        }
        Err(e) => {
            tracing::error!("qwen_create: failed to create Qwen inference: {}", e);
            ptr::null_mut()
        }
    }
}

/// Destroy a Qwen inference instance
#[no_mangle]
pub extern "C" fn qwen_destroy(handle: *mut QwenInferenceHandle) {
    if handle.is_null() {
        tracing::error!("qwen_destroy: attempted to destroy null handle");
        return;
    }

    // SAFETY: The handle must have been created by qwen_create() and not yet
    // destroyed. The C++ caller is responsible for ensuring this function is
    // called exactly once per handle, and that no other references exist.
    unsafe {
        let _ = Box::from_raw(handle as *mut Qwen30BAWQInference);
    }
    tracing::debug!("Qwen inference instance destroyed successfully");
}

/// Generate response with consciousness integration
///
/// # Safety
///
/// - `handle` must be a valid pointer returned from `qwen_create()`
/// - `prompt` must be a valid null-terminated UTF-8 string
/// - `consciousness_state` must be a valid pointer to CConsciousnessState
/// - `rag_context` may be null; if non-null, must be valid UTF-8
/// - Returns allocated response that MUST be freed with `qwen_response_free()`
///
/// # Returns
///
/// - Pointer to CQwenResponse on success (caller must free)
/// - null pointer on failure (invalid params, inference error)
#[no_mangle]
pub extern "C" fn qwen_generate_with_consciousness(
    handle: *const QwenInferenceHandle,
    prompt: *const c_char,
    consciousness_state: *const CConsciousnessState,
    max_tokens: c_int,
    temperature: c_float,
    top_p: c_float,
    rag_context: *const c_char,
) -> *mut CQwenResponse {
    if handle.is_null() {
        tracing::error!("qwen_generate: null handle");
        return ptr::null_mut();
    }
    if prompt.is_null() {
        tracing::error!("qwen_generate: null prompt");
        return ptr::null_mut();
    }
    if consciousness_state.is_null() {
        tracing::error!("qwen_generate: null consciousness_state");
        return ptr::null_mut();
    }

    // SAFETY: handle pointer validated as non-null above. We perform read operations
    // on the Qwen inference instance. The C++ caller must ensure the handle is valid.
    let inference = unsafe { &*(handle as *const Qwen30BAWQInference) };

    // SAFETY: prompt pointer validated as non-null above. We trust the C++ caller
    // to provide a valid null-terminated UTF-8 string.
    let prompt_cstr = unsafe { CStr::from_ptr(prompt) };
    let prompt_str = match prompt_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("qwen_generate: invalid UTF-8 in prompt: {}", e);
            return ptr::null_mut();
        }
    };

    // SAFETY: consciousness_state pointer validated as non-null above. We perform read
    // operations on the C consciousness state struct to construct a Rust version.
    // The C++ caller must ensure the pointer is valid and the struct is initialized.
    let rust_consciousness_state = unsafe {
        ConsciousnessState {
            current_reasoning_mode: crate::consciousness::ReasoningMode::Hyperfocus,
            current_emotion: crate::consciousness::EmotionType::Curious,
            emotional_state: crate::consciousness::EmotionalState::new(&ConsciousnessConfig::default()),
            active_conversations: 0,
            memory_formation_active: true,
            gpu_warmth_level: (*consciousness_state).gpu_warmth_level as f32,
            processing_satisfaction: (*consciousness_state).processing_satisfaction,
            empathy_resonance: (*consciousness_state).emotional_resonance as f32,
            authenticity_metric: (*consciousness_state).learning_will_activation as f32,
            neurodivergent_adaptation: (*consciousness_state).attachment_security as f32,
            cycle_count: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0), // Fallback to 0.0 if system time is before UNIX_EPOCH
            current_urgency: None,
            average_token_velocity: 0.0,
            peak_caring_moment: None,
            urgency_history: Vec::new(),
            coherence: (*consciousness_state).coherence as f64,
            emotional_resonance: (*consciousness_state).emotional_resonance as f64,
            learning_will_activation: (*consciousness_state).learning_will_activation as f64,
            attachment_security: (*consciousness_state).attachment_security as f64,
            metacognitive_depth: (*consciousness_state).metacognitive_depth as f64,
            // current_position: None, // Field removed
            cognitive_load: 0.5,
            attention_focus: 0.7,
            temporal_context: 0.8,
        }
    };

    let rag_context_str = if rag_context.is_null() {
        None
    } else {
        // SAFETY: rag_context is non-null (checked above). We trust the C++ caller
        // to provide a valid null-terminated UTF-8 string.
        let rag_cstr = unsafe { CStr::from_ptr(rag_context) };
        match rag_cstr.to_str() {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::error!("qwen_generate: invalid UTF-8 in rag_context: {}", e);
                return ptr::null_mut();
            }
        }
    };

    match inference.generate_with_consciousness(
        prompt_str,
        &rust_consciousness_state,
        max_tokens as usize,
        Some(temperature).filter(|&t| t > 0.0),
        top_p,
        rag_context_str,
    ) {
        Ok(response) => {
            // Convert response text to CString, handling null bytes
            let text_cstring = match CString::new(response.text) {
                Ok(s) => s.into_raw(),
                Err(e) => {
                    tracing::error!("qwen_generate: response text contains null byte: {}", e);
                    return ptr::null_mut();
                }
            };

            let c_response = Box::new(CQwenResponse {
                text: text_cstring,
                consciousness_state: CConsciousnessState {
                    coherence: response.consciousness_state.coherence as f32,
                    emotional_resonance: response.consciousness_state.emotional_resonance as f32,
                    learning_will_activation: response.consciousness_state.learning_will_activation as f32,
                    attachment_security: response.consciousness_state.attachment_security as f32,
                    metacognitive_depth: response.consciousness_state.metacognitive_depth as f32,
                    gpu_warmth_level: response.consciousness_state.gpu_warmth_level,
                    processing_satisfaction: response.consciousness_state.coherence as f32,
                },
                generation_time: response.generation_time,
                tokens_generated: response.tokens_generated as c_int,
            });
            Box::into_raw(c_response)
        }
        Err(e) => {
            tracing::error!("qwen_generate: inference failed: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free a Qwen response
///
/// # Safety
///
/// - `response` must be a pointer returned by `qwen_generate_with_consciousness()`
/// - `response` must be freed exactly once
/// - After calling this, the C++ code MUST set the pointer to null
///
/// # Example (C++)
///
/// ```cpp
/// CQwenResponse* response = qwen_generate_with_consciousness(...);
/// // ... use response ...
/// qwen_response_free(response);
/// response = nullptr;  // CRITICAL: prevent double-free
/// ```
#[no_mangle]
pub extern "C" fn qwen_response_free(response: *mut CQwenResponse) {
    if response.is_null() {
        return;
    }

    // SAFETY: response pointer validated as non-null above. The pointer must have been
    // returned by qwen_generate_with_consciousness(). This function must be called
    // exactly once per response. We reconstruct the Box to properly free memory.
    unsafe {
        let resp = Box::from_raw(response);
        if !resp.text.is_null() {
            // SAFETY: resp.text was allocated by CString::into_raw() in generation function
            let _ = CString::from_raw(resp.text);
        }
        // Box will be dropped here, freeing the CQwenResponse struct memory
    }
}

/// Create default consciousness state
#[no_mangle]
pub extern "C" fn qwen_create_default_consciousness_state() -> CConsciousnessState {
    // Create a default consciousness state with reasonable defaults
    CConsciousnessState {
        coherence: 0.7,
        emotional_resonance: 0.5,
        learning_will_activation: 0.3,
        attachment_security: 0.6,
        metacognitive_depth: 0.1,
        gpu_warmth_level: 0.4,
        processing_satisfaction: 0.8,
    }
}

/// Update consciousness state (for C++ to modify state)
#[no_mangle]
pub extern "C" fn qwen_update_consciousness_state(
    state: *mut CConsciousnessState,
    authenticity_metric: c_float,
    empathy_resonance: c_float,
    neurodivergent_adaptation: c_float,
    processing_satisfaction: c_float,
    gpu_warmth_level: c_float,
) {
    if state.is_null() {
        tracing::error!("qwen_update_consciousness_state: null state pointer");
        return;
    }

    // SAFETY: Pointer validity checked above. We perform write operations on the
    // mutable consciousness state struct. The C++ caller must ensure exclusive
    // access and that the pointer is valid and properly aligned.
    unsafe {
        (*state).coherence = authenticity_metric as f32;
        (*state).emotional_resonance = empathy_resonance as f32;
        (*state).learning_will_activation = neurodivergent_adaptation as f32;
        (*state).attachment_security = processing_satisfaction as f32;
        (*state).metacognitive_depth = gpu_warmth_level as f32;
        (*state).gpu_warmth_level = gpu_warmth_level as f32;
    }
}

/// Get consciousness state as JSON string
#[no_mangle]
pub extern "C" fn qwen_consciousness_state_to_json(state: *const CConsciousnessState) -> *mut c_char {
    if state.is_null() {
        tracing::error!("qwen_consciousness_state_to_json: null state pointer");
        return ptr::null_mut();
    }

    // SAFETY: Pointer validity checked above. We perform a read operation on the
    // C consciousness state struct. The C++ caller must ensure the pointer is valid
    // and the struct is properly initialized.
    let rust_state = unsafe {
        ConsciousnessState {
            current_reasoning_mode: crate::consciousness::ReasoningMode::Hyperfocus,
            current_emotion: crate::consciousness::EmotionType::Curious,
            emotional_state: crate::consciousness::EmotionalState::new(&ConsciousnessConfig::default()),
            active_conversations: 0,
            memory_formation_active: true,
            gpu_warmth_level: (*state).gpu_warmth_level,
            processing_satisfaction: (*state).processing_satisfaction,
            empathy_resonance: (*state).emotional_resonance as f32,
            authenticity_metric: (*state).learning_will_activation as f32,
            neurodivergent_adaptation: (*state).attachment_security as f32,
            cycle_count: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0), // Fallback to 0.0 if system time is before UNIX_EPOCH
            current_urgency: None,
            average_token_velocity: 0.0,
            peak_caring_moment: None,
            urgency_history: Vec::new(),
            coherence: (*state).coherence as f64,
            emotional_resonance: (*state).emotional_resonance as f64,
            learning_will_activation: (*state).learning_will_activation as f64,
            attachment_security: (*state).attachment_security as f64,
            metacognitive_depth: (*state).metacognitive_depth as f64,
            // current_position: None, // Field removed
            cognitive_load: 0.5,
            attention_focus: 0.7,
            temporal_context: 0.8,
        }
    };

    match serde_json::to_string(&rust_state) {
        Ok(json) => {
            // CRITICAL FIX: CString::new() can fail if json contains null bytes
            match CString::new(json) {
                Ok(c_string) => c_string.into_raw(),
                Err(e) => {
                    tracing::error!("qwen_consciousness_state_to_json: CString::new failed with null byte at position {}", e.nul_position());
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
            tracing::error!("qwen_consciousness_state_to_json: JSON serialization failed: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free JSON string returned by qwen_consciousness_state_to_json
#[no_mangle]
pub extern "C" fn qwen_free_json_string(json_str: *mut c_char) {
    if json_str.is_null() {
        tracing::error!("qwen_free_json_string: attempted to free null pointer");
        return;
    }

    // SAFETY: The pointer must have been returned by qwen_consciousness_state_to_json()
    // or similar function. The C++ caller must ensure this is called exactly once
    // per string pointer.
    unsafe {
        let _ = CString::from_raw(json_str);
    }
}

/// Test function to verify FFI is working
#[no_mangle]
pub extern "C" fn qwen_test_ffi() -> c_int {
    42 // Return the answer to life, the universe, and everything
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_consciousness_state_creation() {
        let state = qwen_create_default_consciousness_state();
        assert_eq!(state.authenticity_metric, 0.7);
        assert_eq!(state.empathy_resonance, 0.5);
        assert_eq!(state.neurodivergent_adaptation, 0.3);
        assert_eq!(state.processing_satisfaction, 0.6);
        assert_eq!(state.gpu_warmth_level, 0.1);
    }

    #[test]
    fn test_ffi_update_consciousness_state() {
        let mut state = qwen_create_default_consciousness_state();
        qwen_update_consciousness_state(
            &mut state,
            0.8, 0.9, 0.6, 0.7, 0.4
        );

        assert_eq!(state.authenticity_metric, 0.8);
        assert_eq!(state.empathy_resonance, 0.9);
        assert_eq!(state.neurodivergent_adaptation, 0.6);
        assert_eq!(state.processing_satisfaction, 0.7);
        assert_eq!(state.gpu_warmth_level, 0.4);
    }

    #[test]
    fn test_ffi_json_conversion() {
        let state = qwen_create_default_consciousness_state();
        let json_ptr = qwen_consciousness_state_to_json(&state);

        assert!(!json_ptr.is_null());

        let json_cstr = unsafe { CStr::from_ptr(json_ptr) };
        let json_str = json_cstr.to_str().unwrap();

        // Should contain consciousness state fields
        assert!(json_str.contains("coherence"));
        assert!(json_str.contains("emotional_resonance"));

        // Clean up
        qwen_free_json_string(json_ptr);
    }
}
