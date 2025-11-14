"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List, Optional
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import io
from PIL import Image

try:
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=['CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace 모델 로드 성공")
except Exception as e:
    print(f"InsightFace 모델 로드 실패: {e}")
    face_app = None

REF_LANDMARKS = np.array([
    [38.2946, 51.6963],  # 왼쪽 눈
    [73.5318, 51.5014],  # 오른쪽 눈
    [56.0252, 71.7366],  # 코 끝
    [41.5493, 92.3655],  # 왼쪽 입꼬리
    [70.7299, 92.2041]   # 오른쪽 입꼬리
], dtype=np.float32)

def _read_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return img_bgr

def _get_embedding_for_image(image_bytes: bytes) -> Optional[np.ndarray]:

    if face_app is None:
        print("모델이 로드되지 않아 임베딩을 추출할 수 없습니다.")
        return None

    img_bgr = _read_image(image_bytes)

    detections = detect_faces(img_bgr)
    
    if not detections:
        print("경고: 얼굴을 탐지하지 못했습니다.")
        return None

    bbox, landmarks = detections[0] 

    M, _ = cv2.estimateAffinePartial2D(landmarks, REF_LANDMARKS, method=cv2.LMEDS)
    
    aligned_face = cv2.warpAffine(img_bgr, M, (112, 112), borderMode=cv2.BORDER_CONSTANT)

    embedding = compute_face_embedding(aligned_face)
    
    return embedding

def detect_faces(image: Any) -> List[Any]:

    if face_app is None:
        raise RuntimeError("InsightFace 모델이 로드되지 않았습니다.")

    bboxes, landmarks = face_app.models['detection'].detect(image, max_num=0, metric='default')

    if bboxes is None or len(bboxes) == 0:
        return []

    results = []
    for i in range(len(bboxes)):
        results.append((bboxes[i], landmarks[i]))

    return results


def compute_face_embedding(face_image: Any) -> Any:

    if face_app is None:
        raise RuntimeError("InsightFace 모델이 로드되지 않았습니다.")

    embedding = face_app.models['recognition'].get_feat(face_image)

    return embedding.flatten()


def detect_face_keypoints(face_image: Any) -> Any:

    # detect_faces 함수에서 keypoints까지 반환함.
    raise NotImplementedError("Student implementation required for keypoint detection")


def warp_face(image: Any, homography_matrix: Any) -> Any:

    # claculate_face_similarity 내부에서 warp를 수행함.
    raise NotImplementedError("Student implementation required for homography warping")


def antispoof_check(face_image: Any) -> float:

    raise NotImplementedError("Student implementation required for face anti-spoofing")


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    emb_a = _get_embedding_for_image(image_a)

    emb_b = _get_embedding_for_image(image_b)

    if emb_a is None or emb_b is None:
        return 0.0

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    
    if norm_a == 0 or norm_b == 0:
        print("경고: 임베딩 벡터의 norm이 0입니다.")
        return 0.0
        
    normalized_emb_a = emb_a / norm_a
    normalized_emb_b = emb_b / norm_b

    similarity = np.dot(normalized_emb_a, normalized_emb_b)

    similarity_score = max(0.0, float(similarity))

    return similarity_score
