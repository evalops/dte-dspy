#!/usr/bin/env python3
"""
Simple test of DSPy with Ollama to debug connection issues.
"""

import dspy
import requests

def test_ollama_connection():
    """Test basic Ollama connection."""
    print("Testing Ollama connection...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ Ollama is running with {len(models)} models")
            for model in models[:3]:  # Show first 3
                print(f"  - {model['name']}")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def test_simple_dspy_ollama():
    """Test simple DSPy call with Ollama."""
    print("\nTesting DSPy with Ollama...")
    
    try:
        # Set up DSPy with Ollama
        lm = dspy.LM(model="ollama/llama3.2:latest", api_base="http://localhost:11434")
        dspy.configure(lm=lm)
        
        print("✓ DSPy configured with Ollama")
        
        # Test simple generation
        print("Testing simple generation...")
        
        class SimpleQA(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()
        
        qa = dspy.Predict(SimpleQA)
        
        # Simple test question
        result = qa(question="What is 2+2?")
        print(f"✓ Generation successful: {result.answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ DSPy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verification_signature():
    """Test our verification signature."""
    print("\nTesting verification signature...")
    
    try:
        class VerifyClaim(dspy.Signature):
            """Decide if a claim is factually correct. Output strictly 'yes' or 'no'."""
            claim = dspy.InputField(desc="The factual claim to verify")
            verdict = dspy.OutputField(desc="either 'yes' or 'no'")
        
        lm = dspy.LM(model="ollama/llama3.2:latest", api_base="http://localhost:11434")
        dspy.configure(lm=lm)
        
        verifier = dspy.Predict(VerifyClaim)
        
        # Test claim
        result = verifier(claim="The capital of France is Paris.")
        print(f"✓ Verification successful: {result.verdict}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== DSPy + Ollama Connection Tests ===")
    
    if not test_ollama_connection():
        return 1
        
    if not test_simple_dspy_ollama():
        return 1
        
    if not test_verification_signature():
        return 1
        
    print("\n✅ All tests passed! DSPy + Ollama is working correctly.")
    return 0

if __name__ == "__main__":
    exit(main())