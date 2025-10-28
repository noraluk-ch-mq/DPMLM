#!/usr/bin/env python3

from DPMLM import DPMLM
import time

def main():
    print("DP-MLM Live Demo")
    print("=" * 50)
    
    print("Loading DP-MLM model...")
    dpmlm = DPMLM()
    print("Model loaded successfully!")
    
    examples = [
        ("I love this movie", 0.5),
        ("This restaurant is amazing", 1.0),
        ("The weather is beautiful today", 2.0)
    ]
    
    print("\nRunning Demo Examples:")
    print("-" * 30)
    
    for i, (text, epsilon) in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"   Input: \"{text}\"")
        print(f"   Privacy level (ε): {epsilon}")
        print("   ⏳ Processing...")
        
        start_time = time.time()
        result = dpmlm.dpmlm_rewrite(text, epsilon)
        end_time = time.time()
        
        if isinstance(result, tuple):
            rewritten_text = result[0]
            changed_words = result[1] if len(result) > 1 else "N/A"
            total_words = result[2] if len(result) > 2 else "N/A"
        else:
            rewritten_text = result
            changed_words = "N/A"
            total_words = "N/A"
        
        print(f"   Output: \"{rewritten_text}\"")
        print(f"   Changed: {changed_words}/{total_words} words")
        print(f"   Time: {end_time - start_time:.2f} seconds")
    
    print("\n" + "=" * 50)
    print("Demo completed! Privacy-preserving text rewriting successful.")

if __name__ == "__main__":
    main()