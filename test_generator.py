"""
Quick test script for OpenAI question generation
"""
import json
import os
from dotenv import load_dotenv
from generator import QuestionGenerator

load_dotenv()

def test_generation():
    """Test basic question generation"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    print("✅ API key found")
    print(f"🚀 Initializing generator...")
    
    generator = QuestionGenerator(api_key, "gpt-4o-mini")
    
    # Test seed question
    seed = {
        "topic": "fill_in_blanks",
        "subtopic": "english_grammar",
        "examSlug": "ssc-cgl",
        "question": "The manager was ____ about the deadline.",
        "options": ["anxious", "anxiety", "anxiously", "anxiousness"],
        "correct": 0
    }
    
    print(f"📝 Testing with seed topic: {seed['topic']}")
    print(f"🎯 Generating 5 questions...")
    
    try:
        response = generator.generate(json.dumps(seed), count=5)
        print(f"✅ Response received ({len(response)} characters)")
        
        # Try to parse
        data = json.loads(response)
        
        if isinstance(data, dict) and 'questions' in data:
            questions = data['questions']
            print(f"✅ Successfully parsed {len(questions)} questions")
            
            # Show first question
            if questions:
                print("\n📋 Sample question:")
                q = questions[0]
                print(f"   QID: {q.get('qid')}")
                print(f"   Question: {q.get('question')[:80]}...")
                print(f"   Options: {q.get('options')}")
                print(f"   Correct: {q.get('correct')}")
                print(f"   Difficulty: {q.get('difficulty')}")
        else:
            print(f"❌ Unexpected format: {type(data)}")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"   Position: {e.pos}")
        print(f"   Message: {e.msg}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_generation()