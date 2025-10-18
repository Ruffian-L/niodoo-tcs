#!/usr/bin/env python3
"""
Test the blog generator directly
"""

from blog_generator import LCARSBlogGenerator

def test_blog_generator():
    print("✍️ Testing Blog Generator directly")
    print("=" * 40)
    
    # Initialize blog generator
    generator = LCARSBlogGenerator()
    
    # Initialize model
    print("🔄 Initializing model...")
    if not generator.initialize_model():
        print("❌ Failed to initialize model")
        return
    
    # Test blog generation
    print("\n🤖 Testing blog generation...")
    topic = "Write a blog post as NiodO.o about helping debug code"
    
    print(f"Topic: {topic}")
    result = generator.generate_blog_post(topic, style="casual", length="medium")
    
    if result:
        print(f"✅ Blog post generated!")
        print(f"Title: {result['title']}")
        print(f"Content length: {result['word_count']} words")
        print(f"Generation time: {result['generation_time']}s")
        print(f"\nContent preview:")
        print("-" * 50)
        print(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
        print("-" * 50)
    else:
        print("❌ Failed to generate blog post")

if __name__ == "__main__":
    test_blog_generator()
