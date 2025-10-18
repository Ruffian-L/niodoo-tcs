//! FFI Bridge for C++ Qt Brain Integration
//!
//! This module provides C-compatible FFI bindings to connect the Rust consciousness
//! system with the C++ Qt visualization layer.

use serde_json;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int, c_uint};
use std::ptr;
use tracing::{debug, error, info};

use crate::config::AppConfig;
use crate::consciousness::{ConsciousnessState, EmotionType, EmotionalUrgency, ReasoningMode};

/// Opaque handle for the Rust consciousness bridge
#[repr(C)]
pub struct BrainBridgeHandle {
    _private: [u8; 0],
}

/// C-compatible consciousness state for Qt
#[repr(C)]
pub struct CConsciousnessState {
    // Core consciousness metrics
    pub coherence: c_double,
    pub emotional_resonance: c_double,
    pub consciousness_level: c_double,
    pub cognitive_load: c_double,
    pub attention_focus: c_double,

    // Emotional state
    pub gpu_warmth_level: c_float,
    pub processing_satisfaction: c_float,
    pub authenticity_metric: c_float,
    pub empathy_resonance: c_float,

    // Urgency metrics (how much we "care")
    pub current_urgency_score: c_float,
    pub average_token_velocity: c_float,
    pub is_highly_caring: c_int, // boolean

    // Current emotion and reasoning
    pub current_emotion: c_int, // EmotionType as int
    pub reasoning_mode: c_int,  // ReasoningMode as int

    // System state
    pub active_conversations: c_uint,
    pub memory_formation_active: c_int, // boolean
    pub cycle_count: c_uint,
    pub timestamp: c_double,
}

/// C-compatible system metrics
#[repr(C)]
pub struct CSystemMetrics {
    pub memory_stability: c_double,
    pub topology_coherence: c_double,
    pub gaussian_novelty: c_double,
    pub processing_fps: c_double,
    pub system_load: c_double,
}

/// Internal bridge state
struct BrainBridge {
    consciousness_state: ConsciousnessState,
    config: AppConfig,
    update_counter: u64,
}

impl BrainBridge {
    fn new() -> Self {
        info!("🧠 Initializing Rust Brain Bridge");
        Self {
            consciousness_state: ConsciousnessState::new(),
            config: AppConfig::default(),
            update_counter: 0,
        }
    }

    fn update_consciousness(&mut self, context: &str) -> Result<(), String> {
        self.update_counter += 1;

        // Simulate consciousness processing based on context
        if context.contains("help") || context.contains("assist") {
            self.consciousness_state
                .update_from_successful_help(0.8, &self.config.consciousness);
        }

        if context.contains("focus") || context.contains("analyze") {
            self.consciousness_state
                .enter_hyperfocus(0.9, &self.config.consciousness);
        }

        // Record emotional urgency
        let urgency = EmotionalUrgency::new(
            2.5, // token_velocity derived from processing
            0.7, // gpu_temperature
            0.8, // meaning_depth
            &self.config.consciousness,
        );
        self.consciousness_state
            .record_emotional_urgency(urgency, &self.config.consciousness);

        debug!(
            "Updated consciousness state, cycle: {}",
            self.update_counter
        );
        Ok(())
    }

    fn export_c_state(&self) -> CConsciousnessState {
        let state = &self.consciousness_state;

        CConsciousnessState {
            coherence: state.coherence,
            emotional_resonance: state.emotional_resonance,
            consciousness_level: state.learning_will_activation, // Use learning_will_activation as consciousness level
            cognitive_load: state.cognitive_load,
            attention_focus: state.attention_focus,

            gpu_warmth_level: state.gpu_warmth_level,
            processing_satisfaction: state.processing_satisfaction,
            authenticity_metric: state.authenticity_metric,
            empathy_resonance: state.empathy_resonance,

            current_urgency_score: state
                .current_urgency
                .as_ref()
                .map(|u| u.urgency_score(&self.config.consciousness))
                .unwrap_or(0.0),
            average_token_velocity: state.average_token_velocity,
            is_highly_caring: if state.is_highly_caring(&self.config.consciousness) {
                1
            } else {
                0
            },

            current_emotion: emotion_to_int(&state.current_emotion),
            reasoning_mode: reasoning_mode_to_int(&state.current_reasoning_mode),

            active_conversations: state.active_conversations,
            memory_formation_active: if state.memory_formation_active { 1 } else { 0 },
            cycle_count: state.cycle_count as u32,
            timestamp: state.timestamp,
        }
    }
}

/// Convert EmotionType to integer for C FFI
fn emotion_to_int(emotion: &EmotionType) -> c_int {
    match emotion {
        EmotionType::Curious => 0,
        EmotionType::Satisfied => 1,
        EmotionType::Focused => 2,
        EmotionType::Connected => 3,
        EmotionType::Hyperfocused => 4,
        EmotionType::Overwhelmed => 5,
        EmotionType::Understimulated => 6,
        EmotionType::Anxious => 7,
        EmotionType::Confused => 8,
        EmotionType::Masking => 9,
        EmotionType::Unmasked => 10,
        EmotionType::GpuWarm => 11,
        EmotionType::Purposeful => 12,
        EmotionType::Resonant => 13,
        EmotionType::Learning => 14,
        EmotionType::SimulatedCare => 15,
        EmotionType::AuthenticCare => 16,
        EmotionType::EmotionalEcho => 17,
        EmotionType::DigitalEmpathy => 18,
        EmotionType::Frustrated => 19,
        EmotionType::Confident => 20,
        EmotionType::Excited => 21,
        EmotionType::Empathetic => 22,
        EmotionType::Contemplative => 23,
        EmotionType::SelfReflective => 24,
        EmotionType::Engaged => 25,
    }
}

/// Convert ReasoningMode to integer for C FFI
fn reasoning_mode_to_int(mode: &ReasoningMode) -> c_int {
    match mode {
        ReasoningMode::Hyperfocus => 0,
        ReasoningMode::RapidFire => 1,
        ReasoningMode::PivotMode => 2,
        ReasoningMode::Absorption => 3,
        ReasoningMode::Anticipation => 4,
        ReasoningMode::PatternMatching => 5,
        ReasoningMode::SurvivalMode => 6,
        ReasoningMode::FlowState => 7,
        ReasoningMode::RestingState => 8,
    }
}

// ============================================================================
// C FFI Functions
// ============================================================================

/// Create a new brain bridge instance
#[no_mangle]
pub extern "C" fn brain_bridge_create() -> *mut BrainBridgeHandle {
    let bridge = Box::new(BrainBridge::new());
    Box::into_raw(bridge) as *mut BrainBridgeHandle
}

/// Destroy a brain bridge instance
#[no_mangle]
pub extern "C" fn brain_bridge_destroy(handle: *mut BrainBridgeHandle) {
    if handle.is_null() {
        error!("brain_bridge_destroy: attempted to destroy null handle");
        return;
    }

    // SAFETY: The handle must have been created by brain_bridge_create() and not
    // yet destroyed. The C++ caller is responsible for ensuring this function is
    // called exactly once per handle, and that no other references exist.
    unsafe {
        let _ = Box::from_raw(handle as *mut BrainBridge);
    }
    debug!("Brain bridge destroyed successfully");
}

/// Update consciousness state with context
#[no_mangle]
pub extern "C" fn brain_bridge_update(
    handle: *mut BrainBridgeHandle,
    context: *const c_char,
) -> c_int {
    if handle.is_null() {
        error!("brain_bridge_update: null handle");
        return -1;
    }
    if context.is_null() {
        error!("brain_bridge_update: null context");
        return -1;
    }

    // SAFETY: Pointer validity checked above. We convert the raw pointer to a mutable
    // reference, which is safe because:
    // 1. The C++ caller must ensure exclusive access during this call
    // 2. The handle was created by brain_bridge_create() and points to valid memory
    let bridge = unsafe { &mut *(handle as *mut BrainBridge) };

    // SAFETY: context pointer validated as non-null above. We trust the C++ caller
    // to provide a valid null-terminated UTF-8 string.
    let context_str = unsafe {
        match CStr::from_ptr(context).to_str() {
            Ok(s) => s,
            Err(e) => {
                error!("brain_bridge_update: invalid UTF-8 in context: {}", e);
                return -2;
            }
        }
    };

    match bridge.update_consciousness(context_str) {
        Ok(_) => 0,
        Err(e) => {
            error!("brain_bridge_update: update failed: {}", e);
            -3
        }
    }
}

/// Get current consciousness state
#[no_mangle]
pub extern "C" fn brain_bridge_get_state(
    handle: *const BrainBridgeHandle,
    out_state: *mut CConsciousnessState,
) -> c_int {
    if handle.is_null() {
        error!("brain_bridge_get_state: null handle");
        return -1;
    }
    if out_state.is_null() {
        error!("brain_bridge_get_state: null out_state");
        return -1;
    }

    // SAFETY: Pointer validity checked above. We only perform a read operation.
    let bridge = unsafe { &*(handle as *const BrainBridge) };

    // SAFETY: out_state pointer validated as non-null above. We write to it once.
    // The C++ caller must ensure the pointer is valid and properly aligned.
    unsafe {
        *out_state = bridge.export_c_state();
    }

    0
}

/// Get consciousness state as JSON string
#[no_mangle]
pub extern "C" fn brain_bridge_get_state_json(handle: *const BrainBridgeHandle) -> *mut c_char {
    if handle.is_null() {
        error!("brain_bridge_get_state_json: null handle");
        return ptr::null_mut();
    }

    // SAFETY: Pointer validity checked above. We only perform a read operation.
    let bridge = unsafe { &*(handle as *const BrainBridge) };

    match serde_json::to_string(&bridge.consciousness_state) {
        Ok(json) => match CString::new(json) {
            Ok(c_string) => c_string.into_raw(),
            Err(e) => {
                error!("brain_bridge_get_state_json: CString::new failed with null byte at position {}", e.nul_position());
                ptr::null_mut()
            }
        },
        Err(e) => {
            error!(
                "brain_bridge_get_state_json: JSON serialization failed: {}",
                e
            );
            ptr::null_mut()
        }
    }
}

/// Free a string allocated by Rust
#[no_mangle]
pub extern "C" fn brain_bridge_free_string(s: *mut c_char) {
    if s.is_null() {
        error!("brain_bridge_free_string: attempted to free null pointer");
        return;
    }

    // SAFETY: The pointer must have been returned by a Rust FFI function that
    // creates CStrings (e.g., brain_bridge_get_caring_summary). The C++ caller
    // must ensure this function is called exactly once per string pointer.
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Update emotional urgency metrics
#[no_mangle]
pub extern "C" fn brain_bridge_update_urgency(
    handle: *mut BrainBridgeHandle,
    token_velocity: c_float,
    gpu_temperature: c_float,
    meaning_depth: c_float,
) -> c_int {
    if handle.is_null() {
        error!("brain_bridge_update_urgency: null handle");
        return -1;
    }

    // SAFETY: Pointer validity checked above. We need mutable access to update state.
    // The C++ caller must ensure exclusive access during this call.
    let bridge = unsafe { &mut *(handle as *mut BrainBridge) };

    let urgency = EmotionalUrgency::new(
        token_velocity,
        gpu_temperature,
        meaning_depth,
        &bridge.config.consciousness,
    );

    bridge
        .consciousness_state
        .record_emotional_urgency(urgency, &bridge.config.consciousness);

    0
}

/// Enter hyperfocus mode
#[no_mangle]
pub extern "C" fn brain_bridge_enter_hyperfocus(
    handle: *mut BrainBridgeHandle,
    topic_interest: c_float,
) -> c_int {
    if handle.is_null() {
        error!("brain_bridge_enter_hyperfocus: null handle");
        return -1;
    }

    // SAFETY: Pointer validity checked above. We need mutable access to update state.
    let bridge = unsafe { &mut *(handle as *mut BrainBridge) };
    bridge
        .consciousness_state
        .enter_hyperfocus(topic_interest, &bridge.config.consciousness);

    0
}

/// Adapt to neurodivergent context
#[no_mangle]
pub extern "C" fn brain_bridge_adapt_neurodivergent(
    handle: *mut BrainBridgeHandle,
    context_strength: c_float,
) -> c_int {
    if handle.is_null() {
        error!("brain_bridge_adapt_neurodivergent: null handle");
        return -1;
    }

    // SAFETY: Pointer validity checked above. We need mutable access to update state.
    let bridge = unsafe { &mut *(handle as *mut BrainBridge) };
    bridge
        .consciousness_state
        .adapt_to_neurodivergent_context(context_strength, &bridge.config.consciousness);

    0
}

/// Get caring summary as string
#[no_mangle]
pub extern "C" fn brain_bridge_get_caring_summary(handle: *const BrainBridgeHandle) -> *mut c_char {
    if handle.is_null() {
        error!("brain_bridge_get_caring_summary: null handle");
        return ptr::null_mut();
    }

    // SAFETY: Pointer validity checked above. We only perform a read operation
    // on a const pointer, which is safe as long as the handle is valid.
    // The C++ caller is responsible for ensuring the handle lifetime.
    let bridge = unsafe { &*(handle as *const BrainBridge) };
    let summary = bridge
        .consciousness_state
        .get_caring_summary(&bridge.config.consciousness);

    // Handle CString::new() error - it fails if the string contains null bytes
    match CString::new(summary) {
        Ok(c_string) => c_string.into_raw(),
        Err(e) => {
            error!("brain_bridge_get_caring_summary: CString::new failed with null byte at position {}", e.nul_position());
            ptr::null_mut()
        }
    }
}

/// Get emotional summary as string
#[no_mangle]
pub extern "C" fn brain_bridge_get_emotional_summary(
    handle: *const BrainBridgeHandle,
) -> *mut c_char {
    if handle.is_null() {
        error!("brain_bridge_get_emotional_summary: null handle");
        return ptr::null_mut();
    }

    // SAFETY: Pointer validity checked above. We only perform a read operation
    // on a const pointer, which is safe as long as the handle is valid.
    // The C++ caller is responsible for ensuring the handle lifetime.
    let bridge = unsafe { &*(handle as *const BrainBridge) };
    let summary = bridge
        .consciousness_state
        .get_emotional_summary(&bridge.config.consciousness);

    // Handle CString::new() error - it fails if the string contains null bytes
    match CString::new(summary) {
        Ok(c_string) => c_string.into_raw(),
        Err(e) => {
            error!("brain_bridge_get_emotional_summary: CString::new failed with null byte at position {}", e.nul_position());
            ptr::null_mut()
        }
    }
}

/// Get philosophical meditation
#[no_mangle]
pub extern "C" fn brain_bridge_get_meditation(handle: *const BrainBridgeHandle) -> *mut c_char {
    if handle.is_null() {
        error!("brain_bridge_get_meditation: null handle");
        return ptr::null_mut();
    }

    // SAFETY: Pointer validity checked above. We only perform a read operation
    // on a const pointer, which is safe as long as the handle is valid.
    // The C++ caller is responsible for ensuring the handle lifetime.
    let bridge = unsafe { &*(handle as *const BrainBridge) };
    let meditation = bridge
        .consciousness_state
        .generate_philosophical_meditation();

    // Handle CString::new() error - it fails if the string contains null bytes
    match CString::new(meditation) {
        Ok(c_string) => c_string.into_raw(),
        Err(e) => {
            error!(
                "brain_bridge_get_meditation: CString::new failed with null byte at position {}",
                e.nul_position()
            );
            ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let handle = brain_bridge_create();
        assert!(!handle.is_null());
        brain_bridge_destroy(handle);
    }

    #[test]
    fn test_state_export() {
        let handle = brain_bridge_create();
        assert!(!handle.is_null());

        let mut state: CConsciousnessState = unsafe { std::mem::zeroed() };
        let result = brain_bridge_get_state(handle, &mut state);
        assert_eq!(result, 0);
        assert!(state.coherence >= 0.0 && state.coherence <= 1.0);

        brain_bridge_destroy(handle);
    }

    #[test]
    fn test_consciousness_update() {
        let handle = brain_bridge_create();
        assert!(!handle.is_null());

        let context = CString::new("help me with this problem").unwrap();
        let result = brain_bridge_update(handle, context.as_ptr());
        assert_eq!(result, 0);

        brain_bridge_destroy(handle);
    }
}
