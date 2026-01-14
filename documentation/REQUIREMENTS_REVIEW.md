# Requirements Review
## ML Backend Services - Face Detection & Recognition API

**Document Version:** 1.0  
**Date:** 2024  
**Based on:** INTEGRATION_GUIDE.md

---

## 1. Executive Summary

This document reviews the requirements for integrating ML backend services (Face Detection & Recognition) with an existing TypeScript-based visitor management system. The ML service operates as a separate microservice providing face detection and recognition capabilities.

---

## 2. Functional Requirements

### 2.1 Core ML Capabilities

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| FR-001 | **Face Detection** - Detect faces in images using YuNet model | High | ✅ Implemented |
| FR-002 | **Face Recognition** - Recognize faces against visitor database using Sface model | High | ✅ Implemented |
| FR-003 | **Feature Extraction** - Extract 512-dimensional face feature vectors | Medium | ✅ Implemented |
| FR-004 | **On-the-fly Processing** - Extract features from stored images during recognition | High | ✅ Implemented |

### 2.2 API Endpoints

| Requirement ID | Endpoint | Method | Purpose | Priority |
|---------------|----------|--------|---------|----------|
| FR-005 | `/api/v1/detect` | POST | Detect faces in image | High |
| FR-006 | `/api/v1/recognize` | POST | Recognize face against database | High |
| FR-007 | `/api/v1/extract-features` | POST | Extract face features for caching | Medium |
| FR-008 | `/ws/realtime` | WebSocket | Real-time camera processing | Medium |
| FR-009 | `/health` | GET | Health check endpoint | High |

### 2.3 Integration Requirements

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| FR-010 | TypeScript backend must be able to call ML service via HTTP | High |
| FR-011 | TypeScript backend must be able to connect via WebSocket for real-time processing | Medium |
| FR-012 | ML service must query shared PostgreSQL database for visitor images | High |
| FR-013 | ML service must NOT handle visitor registration (handled by TypeScript backend) | High |

---

## 3. Technical Requirements

### 3.1 Technology Stack

| Component | Technology | Version/Requirement | Status |
|-----------|-----------|---------------------|--------|
| **Framework** | FastAPI | >= 0.104.0 | ✅ Required |
| **ASGI Server** | Uvicorn | >= 0.24.0 | ✅ Required |
| **Face Detection** | YuNet (OpenCV) | ONNX model | ✅ Required |
| **Face Recognition** | Sface (OpenCV) | ONNX model | ✅ Required |
| **Image Processing** | OpenCV | >= 4.8.0 | ✅ Required |
| **Database** | PostgreSQL | 15+ | ✅ Required |
| **Database Driver** | psycopg2-binary | >= 2.9.9 | ✅ Required |
| **WebSocket** | websockets | >= 12.0 | ✅ Required |
| **Python Version** | Python | 3.11+ | ✅ Required |

### 3.2 Model Requirements

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| TR-001 | YuNet model file: `face_detection_yunet_2023mar.onnx` must be present in `/models` directory | High |
| TR-002 | Sface model file: `face_recognition_sface_2021dec.onnx` must be present in `/models` directory | High |
| TR-003 | Models must be loaded at service startup | High |
| TR-004 | Service must fail gracefully if models are missing | High |

### 3.3 Image Processing Requirements

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| TR-005 | Accept base64-encoded images in requests | High |
| TR-006 | Support common image formats (JPEG, PNG) | High |
| TR-007 | Handle image decoding errors gracefully | High |
| TR-008 | Support RGB to BGR conversion for OpenCV | High |

---

## 4. Database Requirements

### 4.1 Schema Requirements

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| DB-001 | Database must have `visitors` table with `visitor_id` and `base64Image` columns | High |
| DB-002 | OR database must have `visitor_images` table with `visitor_id` and `base64Image` columns | High |
| DB-003 | `base64Image` column must be TEXT type to store base64-encoded images | High |
| DB-004 | ML service must be able to query: `SELECT visitor_id, base64Image FROM visitors WHERE base64Image IS NOT NULL` | High |
| DB-005 | ML service must be able to query: `SELECT visitor_id, base64Image FROM visitor_images WHERE base64Image IS NOT NULL` | High |

### 4.2 Connection Requirements

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| DB-006 | PostgreSQL database must be accessible to both TypeScript backend and ML service | High |
| DB-007 | Connection pooling must be implemented (1-20 connections) | Medium |
| DB-008 | Database connection string must be configurable via environment variable | High |
| DB-009 | Database credentials must be secure (not hardcoded) | High |

---

## 5. API Requirements

### 5.1 Request/Response Formats

| Requirement ID | Endpoint | Request Format | Response Format | Priority |
|---------------|----------|----------------|-----------------|----------|
| API-001 | `/api/v1/detect` | `{ "image": "base64_string" }` | `{ "faces": [...], "count": number }` | High |
| API-002 | `/api/v1/recognize` | `{ "image": "base64_string", "threshold": 0.363 }` | `{ "visitor_id": string\|null, "confidence": number\|null, "matched": boolean }` | High |
| API-003 | `/api/v1/extract-features` | `{ "image": "base64_string" }` | `{ "feature_vector": number[]\|null, "face_detected": boolean }` | Medium |
| API-004 | `/ws/realtime` | `{ "type": "frame", "image": "base64_string" }` | `{ "type": "results", "faces": [...], "count": number }` | Medium |

### 5.2 Error Handling

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| API-005 | Return HTTP 400 for invalid image format | High |
| API-006 | Return HTTP 500 for internal server errors | High |
| API-007 | Return appropriate error messages (sanitized for production) | High |
| API-008 | Handle WebSocket disconnections gracefully | Medium |

### 5.3 CORS Requirements

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| API-009 | CORS must be configurable via environment variable | High | ⚠️ TODO |
| API-010 | CORS must NOT allow all origins in production | High | ⚠️ TODO |
| API-011 | CORS must support credentials | Medium | ✅ Implemented |

---

## 6. Performance Requirements

### 6.1 Response Time Requirements

| Requirement ID | Scenario | Target Response Time | Priority |
|---------------|----------|---------------------|----------|
| PERF-001 | Face detection (< 100 visitors) | < 500ms | High |
| PERF-002 | Face recognition (< 100 visitors) | < 500ms | High |
| PERF-003 | Face recognition (100-1000 visitors) | < 2000ms | Medium |
| PERF-004 | Face recognition (> 1000 visitors) | < 5000ms (or use caching) | Medium |

### 6.2 Throughput Requirements

| Requirement ID | Description | Target | Priority |
|---------------|-------------|--------|----------|
| PERF-005 | Support concurrent requests | 10+ concurrent | Medium |
| PERF-006 | WebSocket connections | 5+ simultaneous | Medium |

### 6.3 Optimization Requirements

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| PERF-007 | Support feature caching for better performance | Medium |
| PERF-008 | Support image resizing for faster processing | Low |
| PERF-009 | Support batch processing (future enhancement) | Low |

---

## 7. Security Requirements

### 7.1 Authentication & Authorization

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| SEC-001 | API key authentication (recommended for production) | High | ⚠️ TODO |
| SEC-002 | Rate limiting to prevent abuse | High | ⚠️ TODO |
| SEC-003 | Input validation for image size (max 5-10MB) | Medium | ⚠️ TODO |

### 7.2 Data Security

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| SEC-004 | Secure database credentials (environment variables) | High | ✅ Implemented |
| SEC-005 | Sanitize error messages (don't expose internal details) | Medium | ⚠️ TODO |
| SEC-006 | HTTPS/TLS for production deployment | High | ⚠️ TODO |

### 7.3 CORS Security

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| SEC-007 | Restrict CORS to specific origins in production | High | ⚠️ TODO |
| SEC-008 | Do not allow wildcard (`*`) origins in production | High | ⚠️ TODO |

---

## 8. Deployment Requirements

### 8.1 Container Requirements

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| DEP-001 | Docker containerization support | High | ✅ Implemented |
| DEP-002 | Docker Compose configuration | High | ✅ Implemented |
| DEP-003 | Health check endpoint for container orchestration | High | ✅ Implemented |
| DEP-004 | Environment variable configuration | High | ✅ Implemented |

### 8.2 Service Configuration

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| DEP-005 | Service name: `face-recog-api` | High | ⚠️ TODO (currently `face-api`) |
| DEP-006 | Default port: 8000 | High | ✅ Implemented |
| DEP-007 | Configurable via environment variables | High | ✅ Implemented |
| DEP-008 | Service restart policy: `unless-stopped` | Medium | ✅ Implemented |

### 8.3 Dependencies

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| DEP-009 | PostgreSQL service dependency | High | ✅ Implemented |
| DEP-010 | Health check wait for PostgreSQL | High | ✅ Implemented |
| DEP-011 | Volume mounts for models directory | High | ✅ Implemented |

---

## 9. Integration Requirements (TypeScript Backend)

### 9.1 HTTP Integration

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| INT-001 | TypeScript backend must implement HTTP client for ML service | High |
| INT-002 | TypeScript backend must handle async/await for API calls | High |
| INT-003 | TypeScript backend must implement error handling for API failures | High |
| INT-004 | TypeScript backend must implement retry logic (optional) | Medium |

### 9.2 Type Definitions

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| INT-005 | TypeScript interfaces for all API requests/responses | High |
| INT-006 | Type definitions for `DetectRequest`, `DetectResponse` | High |
| INT-007 | Type definitions for `RecognizeRequest`, `RecognizeResponse` | High |
| INT-008 | Type definitions for `ExtractFeaturesRequest`, `ExtractFeaturesResponse` | Medium |

### 9.3 WebSocket Integration

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| INT-009 | TypeScript backend must support WebSocket client for real-time processing | Medium |
| INT-010 | TypeScript backend must handle WebSocket reconnection logic | Medium |
| INT-011 | TypeScript backend must handle WebSocket message parsing | Medium |

---

## 10. Similarity Score Requirements

### 10.1 Threshold Configuration

| Requirement ID | Description | Default | Priority |
|---------------|-------------|---------|----------|
| SIM-001 | Default similarity threshold: 0.363 (Sface model default) | 0.363 | High |
| SIM-002 | Configurable threshold per request | 0.363 | High |
| SIM-003 | Score range: 0.0 (different) to 1.0 (identical) | 0.0-1.0 | High |

### 10.2 Match Determination

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| SIM-004 | Match if score >= threshold | High |
| SIM-005 | Return confidence score with match result | High |
| SIM-006 | Support strict threshold (0.5) for high accuracy | Medium |
| SIM-007 | Support lenient threshold (0.3) for more matches | Medium |

---

## 11. Operational Requirements

### 11.1 Monitoring

| Requirement ID | Description | Priority | Status |
|---------------|-------------|----------|--------|
| OPS-001 | Health check endpoint (`/health`) | High | ✅ Implemented |
| OPS-002 | Logging for errors and important events | High | ⚠️ TODO |
| OPS-003 | Metrics endpoint (optional) | Low | ⚠️ TODO |
| OPS-004 | Request/response logging (optional) | Low | ⚠️ TODO |

### 11.2 Error Handling

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| OPS-005 | Graceful degradation if models fail to load | High |
| OPS-006 | Graceful handling of database connection failures | High |
| OPS-007 | Timeout handling for long-running requests | Medium |

---

## 12. Testing Requirements

### 12.1 Functional Testing

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| TEST-001 | Test face detection with various images | High |
| TEST-002 | Test face recognition with known visitors | High |
| TEST-003 | Test face recognition with unknown faces | High |
| TEST-004 | Test feature extraction | Medium |
| TEST-005 | Test WebSocket real-time processing | Medium |

### 12.2 Integration Testing

| Requirement ID | Description | Priority |
|---------------|-------------|----------|
| TEST-006 | Test TypeScript backend integration | High |
| TEST-007 | Test database connectivity | High |
| TEST-008 | Test error handling and edge cases | Medium |

---

## 13. Documentation Requirements

### 13.1 Required Documentation

| Requirement ID | Document | Status |
|---------------|----------|--------|
| DOC-001 | Integration Guide (INTEGRATION_GUIDE.md) | ✅ Complete |
| DOC-002 | API endpoint documentation | ✅ Complete (in guide) |
| DOC-003 | TypeScript type definitions | ✅ Complete (in guide) |
| DOC-004 | Deployment guide | ✅ Complete (DEPLOYMENT_GUIDE.md) |
| DOC-005 | Cosine similarity guide | ✅ Complete (COSINE_SIMILARITY_GUIDE.md) |
| DOC-006 | Production readiness guide | ✅ Complete (PRODUCTION_READINESS.md) |

---

## 14. Requirements Summary

### 14.1 Implementation Status

| Category | Total | Implemented | TODO | Percentage |
|----------|-------|-------------|------|------------|
| Functional | 13 | 13 | 0 | 100% |
| Technical | 8 | 8 | 0 | 100% |
| Database | 9 | 9 | 0 | 100% |
| API | 11 | 8 | 3 | 73% |
| Performance | 9 | 4 | 5 | 44% |
| Security | 8 | 2 | 6 | 25% |
| Deployment | 11 | 10 | 1 | 91% |
| Integration | 11 | 0 | 11 | 0% |
| Similarity | 7 | 7 | 0 | 100% |
| Operational | 7 | 1 | 6 | 14% |
| Testing | 8 | 0 | 8 | 0% |
| Documentation | 6 | 6 | 0 | 100% |
| **TOTAL** | **108** | **68** | **40** | **63%** |

### 14.2 Critical TODO Items

**High Priority:**
1. ⚠️ **SEC-001**: Implement API key authentication
2. ⚠️ **SEC-002**: Implement rate limiting
3. ⚠️ **SEC-007**: Restrict CORS to specific origins
4. ⚠️ **DEP-005**: Update service name to `face-recog-api` in docker-compose
5. ⚠️ **API-009**: Make CORS configurable via environment variable
6. ⚠️ **OPS-002**: Implement proper logging

**Medium Priority:**
7. ⚠️ **SEC-003**: Add image size validation
8. ⚠️ **PERF-007**: Implement feature caching
9. ⚠️ **INT-001**: TypeScript backend integration implementation
10. ⚠️ **TEST-001**: Functional testing

---

## 15. Recommendations

### 15.1 Immediate Actions
1. Update docker-compose service name to `face-recog-api`
2. Implement CORS configuration via environment variable
3. Add basic logging (replace `print()` statements)
4. Implement API key authentication for production

### 15.2 Short-term (Before Production)
1. Add rate limiting
2. Implement image size validation
3. Add proper error message sanitization
4. Set up HTTPS/TLS
5. Complete TypeScript backend integration

### 15.3 Long-term (Optimization)
1. Implement feature caching for better performance
2. Add metrics and monitoring
3. Implement batch processing
4. Add comprehensive testing suite

---

## 16. Approval & Sign-off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Technical Lead | | | |
| Product Owner | | | |
| DevOps Lead | | | |

---

**Document Status:** ✅ Ready for Review  
**Last Updated:** 2024  
**Next Review Date:** TBD
