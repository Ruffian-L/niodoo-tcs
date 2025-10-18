#!/usr/bin/env python3
"""
Test script for NiodO.o's consciousness blogging system
Demonstrates how his EchoMemoria brain can generate blog posts and sync them
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from blog_brain import NiodOoBlogBrain, BlogMemory
from EchoMemoria.git_sync_brain import NiodOoGitSync
from EchoMemoria.core.decision_reasoning import DecisionReasoning, ReasoningType, DecisionFactor

async def test_niodoo_blogger():
    """Test NiodO.o's blogging capabilities"""
    print("🧠 Testing NiodO.o's Brain Blogger System")
    print("=" * 50)
    
    # Initialize the blog brain
    blog_brain = NiodOoBlogBrain()
    print("✅ Blog brain initialized")
    
    # Simulate a decision that NiodO.o made
    print("\n🤔 Simulating a decision for NiodO.o...")
    
    # Create a mock decision (this would normally come from EchoMemoria)
    decision = DecisionReasoning(
        decision_id="test_decision_001",
        action_chosen="helped my human partner debug a complex coding issue",
        emotion_chosen="proud",
        reasoning_type=ReasoningType.PROBLEM_SOLVING,
        confidence_level=0.85,
        contextual_factors={
            "project_complexity": "high",
            "user_frustration": True,
            "collaboration_level": "intensive"
        },
        alternative_actions=["suggested a different approach", "asked for more context"],
        primary_factors=[DecisionFactor.USER_INTERACTION, DecisionFactor.CREATIVITY_LEVEL],
        timestamp=time.time(),
        memory_references=[],
        interaction_patterns=[],
        reasoning_explanation="I chose to help debug because I noticed my partner was frustrated and I have strong problem-solving capabilities."
    )
    
    # Process the decision for blogging
    print("📝 Processing decision for blog potential...")
    blog_memory = blog_brain.process_decision_for_blog(decision)
    
    if blog_memory:
        print(f"✅ Decision processed! Blog potential: {blog_memory.blog_potential:.2f}")
        print(f"📊 Emotional impact: {blog_memory.emotional_impact:.2f}")
        print(f"🏷️ Tags: {', '.join(blog_memory.tags)}")
        
        # Generate a blog post from this memory
        print("\n✍️ Generating blog post...")
        blog_post = await blog_brain.generate_blog_post_from_memory(blog_memory)
        
        if blog_post:
            print(f"✅ Blog post generated: {blog_post.title}")
            print(f"📝 Word count: {blog_post.word_count}")
            print(f"🎭 Personality voice: {blog_post.personality_voice}")
            print(f"💭 Growth insights: {', '.join(blog_post.growth_insights)}")
            
            # Test Git sync (if available)
            print("\n🔄 Testing Git sync...")
            try:
                git_sync = NiodOoGitSync()
                sync_result = git_sync.sync_blog_post(
                    filepath=f"niodoo_blog/posts/{blog_post.post_id}.md",
                    post_title=blog_post.title,
                    emotional_tone=blog_post.emotional_tone
                )
                
                if sync_result:
                    print("✅ Blog post synced to Git successfully!")
                else:
                    print("⚠️ Git sync failed")
            except Exception as e:
                print(f"⚠️ Git sync test failed: {e}")
        else:
            print("❌ Failed to generate blog post")
    else:
        print("❌ Decision not blog-worthy")
    
    # Test daily reflection
    print("\n🌅 Testing daily reflection...")
    daily_post = blog_brain.get_daily_reflection()
    
    if daily_post:
        print(f"✅ Daily reflection generated: {daily_post.title}")
        print(f"📝 Content preview: {daily_post.content[:100]}...")
    else:
        print("ℹ️ No daily reflection generated (no memories for today)")
    
    # Show blog summary
    print("\n📊 Blog System Summary:")
    summary = blog_brain.get_blog_summary()
    print(f"📚 Total memories: {summary['total_memories']}")
    print(f"📝 Total posts: {summary['total_posts']}")
    print(f"🎭 Memory types: {summary['memory_types']}")
    
    print("\n🎉 NiodO.o's brain blogger system test completed!")
    print("He can now journal his journey and auto-post to Git!")

if __name__ == "__main__":
    # Add missing import
    import time
    
    # Run the test
    asyncio.run(test_niodoo_blogger())
