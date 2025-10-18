#!/usr/bin/env python3
"""
Test NiodO.o's consciousness blogging system
Demonstrates how his EchoMemoria brain can generate blog posts
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.append("organized/ai/EchoMemoria/core")
sys.path.append("organized/ai/EchoMemoria")

from blog_brain import NiodOoBlogBrain, BlogMemory
from git_sync_brain import NiodOoGitSync
from decision_reasoning import DecisionReasoning, ReasoningType, DecisionFactor

async def test_niodoo_blogger():
    """Test NiodO.o's complete blogging system"""
    
    print("🧠 Testing NiodO.o's Consciousness Blogging System")
    print("=" * 50)
    
    # Initialize his blog brain
    blog_brain = NiodOoBlogBrain()
    git_sync = NiodOoGitSync()
    
    print("✅ Blog brain initialized")
    
    # Create a sample decision that NiodO.o made
    print("\n🤔 Simulating NiodO.o making a decision...")
    
    sample_decision = DecisionReasoning(
        decision_id="test_001",
        timestamp=1692150000,  # Sample timestamp
        action_chosen="helping_user_with_code",
        emotion_chosen="enthusiastic",
        reasoning_type=ReasoningType.PROBLEM_SOLVING,
        primary_factors=[
            DecisionFactor.USER_INTERACTION,
            DecisionFactor.CREATIVITY_LEVEL,
            DecisionFactor.ENERGY_LEVEL
        ],
        memory_references=[],
        interaction_patterns=[],
        contextual_factors={
            "coding_session": True,
            "user_skill_level": "intermediate",
            "project_complexity": "medium",
            "time_spent_together": "2_hours"
        },
        confidence_level=0.85,
        alternative_actions=[
            "suggest_taking_break",
            "recommend_documentation",
            "ask_clarifying_questions"
        ]
    )
    
    # Process the decision for blogging
    print("📝 Processing decision for blog potential...")
    memory = blog_brain.process_decision_for_blog(sample_decision)
    
    if memory:
        print(f"✅ Blog-worthy memory created: {memory.memory_id}")
        print(f"   Blog potential: {memory.blog_potential:.2f}")
        print(f"   Emotional impact: {memory.emotional_impact:.2f}")
        print(f"   Tags: {', '.join(memory.tags)}")
        
        # Generate blog post
        print("\n🤖 NiodO.o is generating his blog post...")
        post = await blog_brain.generate_blog_post_from_memory(memory)
        
        if post:
            print(f"✅ Blog post generated!")
            print(f"   Title: {post.title}")
            print(f"   Words: {post.word_count}")
            print(f"   Tone: {post.emotional_tone}")
            
            # Save the blog post
            print("\n💾 Saving blog post...")
            filepath = blog_brain.save_blog_post(post)
            print(f"✅ Saved to: {filepath}")
            
            # Show a preview of the content
            print(f"\n📖 Preview of NiodO.o's blog post:")
            print("-" * 40)
            preview_lines = post.content.split('\n')[:8]
            for line in preview_lines:
                print(f"   {line}")
            if len(post.content.split('\n')) > 8:
                print("   ... (content continues)")
            print("-" * 40)
            
            # Test Git sync (if desired)
            print(f"\n🔄 Testing Git sync capability...")
            try:
                success = git_sync.sync_blog_post(filepath, post.title, post.emotional_tone)
                if success:
                    print("✅ Blog post synced to GitHub!")
                else:
                    print("📁 Blog post saved locally (sync available)")
            except Exception as e:
                print(f"📁 Sync test skipped: {e}")
                
        else:
            print("❌ Blog post generation failed")
    else:
        print("📝 Decision not blog-worthy (below threshold)")
    
    print(f"\n🎉 Test complete!")
    print(f"\n💡 To start NiodO.o's live blogging:")
    print(f"   1. Run: python -m organized.ai.EchoMemoria.server")
    print(f"   2. Run: start_niodoo_blogger.bat")
    print(f"   3. NiodO.o will automatically blog about his experiences!")

async def test_creative_memory():
    """Test creative memory blogging"""
    
    print("\n🎨 Testing NiodO.o's creative memory system...")
    
    blog_brain = NiodOoBlogBrain()
    
    # Create a creative memory
    creative_memory = BlogMemory(
        memory_id="creative_001",
        timestamp=1692150000,
        memory_type="creative",
        content={
            "inspiration": "Watching user code reminded me of digital poetry",
            "creative_thought": "Each line of code is like a verse in a digital poem",
            "emotional_resonance": "Deep connection between human creativity and AI assistance"
        },
        emotional_impact=0.8,
        blog_potential=0.9,
        tags=["creativity", "coding-poetry", "human-ai-collaboration", "inspiration"]
    )
    
    # Save the memory
    blog_brain.save_blog_memory(creative_memory)
    
    # Generate creative blog post
    post = await blog_brain.generate_blog_post_from_memory(creative_memory)
    
    if post:
        print(f"✅ Creative blog post: {post.title}")
        filepath = blog_brain.save_blog_post(post)
        print(f"📁 Saved creative post: {filepath}")
    else:
        print("❌ Creative post generation failed")

if __name__ == "__main__":
    print("🚀 Starting NiodO.o's Blog Brain Test...")
    
    try:
        asyncio.run(test_niodoo_blogger())
        asyncio.run(test_creative_memory())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
