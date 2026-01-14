# Cosine Similarity in Face Recognition

## Overview

The Sface model uses **cosine similarity** to compare face feature vectors. This guide explains how it works and how to interpret the scores.

## What is Cosine Similarity?

Cosine similarity measures the **angle** between two vectors in high-dimensional space. For face recognition:
- Each face is represented as a **512-dimensional feature vector**
- Cosine similarity measures how "similar" the directions of two vectors are
- Range: **-1.0 to 1.0** (for normalized vectors, typically **0.0 to 1.0**)

## How It's Calculated

### Mathematical Formula

```
cosine_similarity = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product of vectors A and B
- `||A||` = magnitude (length) of vector A
- `||B||` = magnitude (length) of vector B

### In OpenCV Sface

OpenCV's `FaceRecognizerSF.match()` with `FR_COSINE` mode calculates:

```python
score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
```

This returns a **similarity score** where:
- **Higher score = More similar faces**
- **Lower score = Less similar faces**

## Score Interpretation

### Score Range

For Sface model, typical scores are:
- **0.0 - 0.3**: Very different faces (different people)
- **0.3 - 0.5**: Somewhat similar (might be same person, but uncertain)
- **0.5 - 0.7**: Similar faces (likely same person)
- **0.7 - 1.0**: Very similar faces (same person with high confidence)

### Default Threshold

The Sface model uses a **default threshold of 0.363**:
- **score ≥ 0.363**: Considered a **MATCH** (same person)
- **score < 0.363**: Considered **NO MATCH** (different person)

## Determining Matches

### Basic Matching Logic

```python
def compare_faces(feature1, feature2, threshold=0.363):
    score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    is_match = score >= threshold
    return score, is_match
```

### Example Scores

| Score | Interpretation | Match? |
|-------|---------------|--------|
| 0.85 | Very high similarity | ✅ **MATCH** |
| 0.65 | High similarity | ✅ **MATCH** |
| 0.45 | Moderate similarity | ✅ **MATCH** (above threshold) |
| 0.363 | Exactly at threshold | ✅ **MATCH** (borderline) |
| 0.35 | Just below threshold | ❌ **NO MATCH** |
| 0.25 | Low similarity | ❌ **NO MATCH** |
| 0.10 | Very low similarity | ❌ **NO MATCH** |

## Adjusting the Threshold

### Lower Threshold (e.g., 0.3)
- **More lenient**: Accepts more matches
- **Higher false positive rate**: Might match different people
- **Use when**: You want to catch all possible matches, even if some are wrong

### Higher Threshold (e.g., 0.5)
- **More strict**: Only accepts very similar faces
- **Lower false positive rate**: More confident matches
- **Use when**: You want high accuracy, even if you miss some matches

### Recommended Thresholds

```python
# Very strict (high accuracy, might miss some matches)
threshold = 0.5

# Default Sface threshold (balanced)
threshold = 0.363

# Lenient (catches more matches, might have false positives)
threshold = 0.3
```

## Real-World Examples

### Example 1: Same Person, Different Conditions

```
Person A (registration photo) vs Person A (camera, different lighting)
Score: 0.72
Result: ✅ MATCH (high confidence)
```

### Example 2: Same Person, Similar Conditions

```
Person A (registration) vs Person A (camera, similar lighting)
Score: 0.58
Result: ✅ MATCH (good confidence)
```

### Example 3: Different People

```
Person A (registration) vs Person B (camera)
Score: 0.28
Result: ❌ NO MATCH (correctly rejected)
```

### Example 4: Borderline Case

```
Person A (registration) vs Person A (camera, poor quality)
Score: 0.38
Result: ✅ MATCH (with threshold=0.363, but close call)
```

## Factors Affecting Scores

### Higher Scores (Better Matches)
- ✅ Same person
- ✅ Similar lighting conditions
- ✅ Similar face angle/pose
- ✅ Good image quality
- ✅ Recent photos (less aging)

### Lower Scores (Worse Matches)
- ❌ Different people
- ❌ Very different lighting
- ❌ Different face angles
- ❌ Poor image quality
- ❌ Significant time difference (aging)
- ❌ Facial changes (glasses, beard, makeup)

## Best Practices

### 1. Use Multiple Images for Registration
```python
# Extract features from 5-10 images
# Average the features for better accuracy
avg_feature = np.mean([feature1, feature2, feature3, ...], axis=0)
```

### 2. Adjust Threshold Based on Use Case
```python
# High-security: Use higher threshold (0.5)
# General access: Use default (0.363)
# Lenient matching: Use lower threshold (0.3)
```

### 3. Consider Confidence Levels
```python
score, is_match = compare_faces(feature1, feature2, threshold=0.363)

if score >= 0.7:
    confidence = "Very High"
elif score >= 0.5:
    confidence = "High"
elif score >= 0.363:
    confidence = "Medium"
else:
    confidence = "Low"
```

### 4. Handle Edge Cases
```python
# If score is very close to threshold, request additional verification
if 0.35 <= score < 0.363:
    # Borderline case - might want to ask for re-scan
    return "UNCERTAIN", score
```

## Code Example

```python
import cv2
import numpy as np

# Initialize Sface recognizer
recognizer = cv2.FaceRecognizerSF.create(
    model='face_recognition_sface_2021dec.onnx',
    config='',
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Extract features from two faces
feature1 = recognizer.feature(face1_image)
feature2 = recognizer.feature(face2_image)

# Calculate cosine similarity
score = recognizer.match(
    feature1,
    feature2,
    cv2.FaceRecognizerSF_FR_COSINE
)

# Determine match
threshold = 0.363
is_match = score >= threshold

print(f"Similarity Score: {score:.3f}")
print(f"Threshold: {threshold}")
print(f"Match: {is_match}")

if is_match:
    if score >= 0.7:
        print("✅ High confidence match")
    elif score >= 0.5:
        print("✅ Good match")
    else:
        print("✅ Match (above threshold)")
else:
    print("❌ No match")
```

## Summary

- **Cosine similarity** measures how similar two face feature vectors are
- **Score range**: 0.0 (different) to 1.0 (identical)
- **Default threshold**: 0.363 for Sface model
- **Higher score = More similar faces**
- **score ≥ threshold = MATCH**
- Adjust threshold based on your accuracy vs. coverage needs
