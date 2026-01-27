# Guide to Cosine Similarity for Face Recognition

## Overview

Cosine similarity is the core metric for comparing face feature vectors in the VMS backend, specifically via the OpenCV SFace model. This guide details how cosine similarity works, how it's used in our stack, and best practices for threshold tuning.

## What is Cosine Similarity?

Cosine similarity quantifies the angle between two vectors in high-dimensional space:

- Each face is extracted as a **512-dimensional feature vector** using the SFace ONNX model
- Similar vectors (smaller angles) mean more similar faces
- Range for normalized features: **0.0** (completely different) to **1.0** (identical)

## How Cosine Similarity is Calculated

**Mathematical Formula:**

```
cosine_similarity = (A · B) / (||A|| × ||B||)
```
Where:
- `A · B`: Dot product of vectors A and B
- `||A||` and `||B||`: Euclidean norms of A and B

**In This Project (with OpenCV SFace):**

```python
score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
```
A higher `score` means faces are more likely to be the same person.

## Score Ranges & Interpretation

| Score  | Meaning                  | Guidance         |
| ------ | ------------------------ | ---------------- |
| 0.0–0.3  | Unrelated faces           | Always NO MATCH  |
| 0.3–0.5  | Some similarity (borderline cases possible) | Use threshold |
| 0.5–0.7  | Strong similarity         | Likely MATCH     |
| 0.7–1.0  | Very high similarity      | High-confidence MATCH |

### Default Matching Threshold

- **score ≥ 0.363**: **MATCH** (same person, as per SFace default)
- **score < 0.363**: **NO MATCH** (different people)

## Python Matching Logic (as used in backend services)

```python
def compare_faces(feature1, feature2, threshold=0.363):
    score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    is_match = score >= threshold
    return score, is_match
```

**Example Score Table:**  
| Score | Interpretation            | Match?         |
|-------|--------------------------|----------------|
| 0.85  | Very high similarity     | ✅ MATCH       |
| 0.65  | High similarity          | ✅ MATCH       |
| 0.45  | Moderate similarity      | ✅ MATCH (above threshold) |
| 0.363 | At threshold             | ✅ MATCH (borderline) |
| 0.35  | Just below threshold     | ❌ NO MATCH    |
| 0.25  | Low similarity           | ❌ NO MATCH    |
| 0.10  | Unrelated                | ❌ NO MATCH    |

## Adjusting the Threshold in VMS

Thresholds can be raised or lowered depending on application needs:

- **Lower (e.g. 0.3):** More matches, increased risk of false positives
- **Higher (e.g. 0.5):** Fewer matches, higher accuracy, may omit valid matches

**Recommended thresholds for this project:**
```python
threshold = 0.5   # Strict (high security)
threshold = 0.363 # Default (balanced, as in SFace)
threshold = 0.3   # Lenient (catch-all, higher false positives)
```

## Real-World Recognition Examples

- **Same person – different lighting:**  
  Score: 0.72 → ✅ MATCH (high confidence)
- **Same person – similar angle:**  
  Score: 0.58 → ✅ MATCH (good confidence)
- **Different people:**  
  Score: 0.28 → ❌ NO MATCH
- **Borderline quality (noise/blur):**  
  Score: 0.38 → ✅ MATCH (but double-check, near threshold)

## Factors That Affect Cosine Similarity Scores

- **High scores:** Same identity, good lighting, similar angle, high image quality, recent photos
- **Low scores:** Different faces, or same person under different angles, with poor image quality, occlusions, aging, or changes such as facial hair/glasses

## Implementation Best Practices for VMS

**1. Register Multiple Photos Per Person**
```python
avg_feature = np.mean([feature1, feature2, ...], axis=0)  # Improves robustness
```

**2. Set Threshold Per Use-Case**

- High security area: 0.5 or above
- General access: default 0.363
- Flexible/QA/testing: 0.3

**3. Handle Confidence Levels in UI/Logic**
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

**4. Address Borderline Cases in User Experience**
```python
# Suggest additional scan if borderline
if 0.35 <= score < 0.363:
    return "UNCERTAIN", score
```

## End-to-End Example (for VMS Python ML Backend)

```python
import cv2
import numpy as np

recognizer = cv2.FaceRecognizerSF.create(
    model='face_recognition_sface_2021dec.onnx',
    config='',
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU,
)

feature1 = recognizer.feature(face1_image)
feature2 = recognizer.feature(face2_image)
score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)

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

- Cosine similarity is the official vector metric for SFace-based recognition in VMS
- **Default threshold 0.363**; adjust as needed for security or tolerance
- Use multiple images per person for stronger embedding
- Always support threshold/score-driven feedback in UX for edge cases
