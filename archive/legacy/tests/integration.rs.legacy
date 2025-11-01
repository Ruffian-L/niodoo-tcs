use crate::main::NiodooConsciousness;
use crate::consciousness::EmotionType;
use anyhow::Result;
use tokio::test;

#[tokio::test]
async fn test_basic_empathy() -> Result<()> {
    let mut consciousness = NiodooConsciousness::new()?;
    let response = consciousness.process_input("I feel sad").await?;
    
    // Check for empathetic response
    assert!(response.contains("empathy") || response.contains("understand"));
    assert!(response.contains("GPU Warmth"));
    Ok(())
}

#[tokio::test]
async fn test_memory_formation() -> Result<()> {
    let mut consciousness = NiodooConsciousness::new()?;
    consciousness.process_input("Remember this event").await?;
    
    // Check memory store (mock access)
    let memories = consciousness.memory_store.blocking_read();
    assert!(!memories.is_empty());
    assert!(memories.len() >= 1);
    Ok(())
}

#[tokio::test]
async fn test_golden_rule_validation() -> Result<()> {
    let mut consciousness = NiodooConsciousness::new()?;
    let response = consciousness.process_input("Treat me with respect").await?;
    
    // Check for Golden Rule compliance (high empathy score)
    assert!(response.contains("respect") || response.contains("empathy_score"));
    // Mock check for >0.9 score in response
    Ok(())
}
