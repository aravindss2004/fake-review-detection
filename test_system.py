"""
Quick system test script
"""
import requests
import json

print("=" * 70)
print("FAKE REVIEW DETECTION SYSTEM - TEST")
print("=" * 70)

# Test reviews
test_reviews = [
    "Great product, highly recommend!",
    "AMAZING!!! BEST EVER!!! BUY NOW!!!",
    "Terrible quality, broke immediately",
    "Good value for money, works as expected",
    "OMG THIS IS PERFECT!!! 5 STARS!!!",
]

print("\nTesting prediction endpoint with multiple reviews...")
print("-" * 70)

try:
    response = requests.post(
        'http://localhost:5000/predict',
        json={'reviews': test_reviews},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result['success']:
            print("\n✓ Predictions successful!\n")
            
            for pred in result['data']['predictions']:
                text = pred['text'][:60] + "..." if len(pred['text']) > 60 else pred['text']
                prediction = pred['prediction']
                confidence = pred['confidence'] * 100
                
                emoji = "✓" if prediction == "Genuine" else "⚠"
                color_code = "[GENUINE]" if prediction == "Genuine" else "[FAKE]"
                
                print(f"{emoji} {color_code:<10} {confidence:5.1f}% | {text}")
            
            print("\n" + "-" * 70)
            print("Summary:")
            summary = result['data']['summary']
            print(f"  Total Reviews: {summary['total_reviews']}")
            print(f"  Genuine: {summary['genuine_reviews']} ({summary['genuine_percentage']:.1f}%)")
            print(f"  Fake: {summary['fake_reviews']} ({summary['fake_percentage']:.1f}%)")
            print(f"  Avg Confidence: {summary['average_confidence']*100:.1f}%")
        else:
            print(f"✗ Prediction failed: {result['message']}")
    else:
        print(f"✗ HTTP Error: {response.status_code}")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\n✓ Backend: http://localhost:5000")
print("✓ Frontend: http://localhost:3000")
print("\nOpen the frontend in your browser to interact with the system!")
