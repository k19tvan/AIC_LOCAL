import os
import sys
import time
import traceback
import base64
import asyncio
import operator
import json
import re
import uuid
import shutil
from pathlib import Path
from collections import defaultdict
import httpx
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, ORJSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, ValidationError
from pymilvus import Collection, connections, utility
import requests
try:
    from pymilvus import LoadState as _MilvusLoadState
except ImportError:
    try:
        from pymilvus.client.constants import LoadState as _MilvusLoadState
    except Exception:
        _MilvusLoadState = None
from elasticsearch import Elasticsearch, NotFoundError
import polars as pl
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote

import cv2
try:
    import av
    PYAV_AVAILABLE = True
    print("--- pyav library successfully imported. AV1 support enabled. ---")
except ImportError:
    PYAV_AVAILABLE = False
    print("!!! WARNING: pyav library not found. AV1 support is disabled. pip install av !!!")

# --- PERFORMANCE OPTIMIZATION: Caches ---
embedding_cache = {}
processed_query_cache = {}
CACHE_MAX_SIZE = 500
# -----------------------------------------

# ## START: GOOGLE IMAGE SEARCH INTEGRATION (HELPERS & MODELS) ##
google_search_session = requests.Session()
google_search_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
})

def get_google_images(keyword: str, k: int = 15):
    try:
        url = f"https://www.google.com/search?q={quote(keyword)}&tbm=isch"
        html = google_search_session.get(url, timeout=15).text
        start = html.find('["https://')
        if start == -1:
            return []
        html = html[start:]
        image_links = re.findall(r'\["(https?://[^"]+)",\d+,\d+]', html)
        seen = set()
        results = []
        for link in image_links:
            if not link.startswith("https://encrypted-tbn0.gstatic.com") and link not in seen:
                seen.add(link)
                results.append(link)
                if len(results) >= k:
                    break
        return results
    except Exception as e:
        print(f"Error during Google Image Search: {e}")
        return []

class GoogleImageSearchRequest(BaseModel):
    query: str

class DownloadImageRequest(BaseModel):
    url: str
# ## END: GOOGLE IMAGE SEARCH INTEGRATION (HELPERS & MODELS) ##


# --- Thiết lập & Cấu hình ---
app = FastAPI(default_response_class=ORJSONResponse)

origins = ["http://localhost:2108"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:2108"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = Path("/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_BASE_DIR = "/workspace/mlcv2"

# ## START: GOOGLE IMAGE SEARCH API ENDPOINTS ##
@app.post("/google_image_search")
async def google_image_search(request_data: GoogleImageSearchRequest):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        image_urls = await asyncio.to_thread(get_google_images, request_data.query)
        return {"image_urls": image_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search Google Images: {e}")

@app.post("/download_external_image")
async def download_external_image(request_data: DownloadImageRequest):
    try:
        response = requests.get(request_data.url, stream=True, timeout=20, headers=google_search_session.headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', 'image/jpeg')
        if 'image' not in content_type:
            raise HTTPException(status_code=400, detail="URL does not point to a valid image.")
            
        extension = ".jpg"
        if 'png' in content_type: extension = '.png'
        elif 'webp' in content_type: extension = '.webp'
        elif 'gif' in content_type: extension = '.gif'

        temp_filename = f"g-search-{uuid.uuid4()}{extension}"
        temp_filepath = TEMP_UPLOAD_DIR / temp_filename

        with temp_filepath.open("wb") as buffer:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)

        full_path_str = str(temp_filepath.resolve())
        
        return {
            "temp_image_name": temp_filename,
            "filepath": full_path_str,
            "url": f"/images/{base64.urlsafe_b64encode(full_path_str.encode('utf-8')).decode('utf-8')}"
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download image from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during image download: {e}")
# ## END: GOOGLE IMAGE SEARCH API ENDPOINTS ##

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory="static"), name="static")

_CURRENT_DIR_PARENT = os.path.dirname(os.path.abspath(__file__))
COMMON_PARENT_DIR = os.path.dirname(_CURRENT_DIR_PARENT)
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)

try:
    from function import translate_query, enhance_query, expand_query_parallel
    print("--- Gateway Server: Đã import thành công các hàm xử lý truy vấn. ---")
except ImportError:
    print("!!! CẢNH BÁO: Không thể import các hàm xử lý truy vấn. Sử dụng hàm DUMMY. !!!")
    def enhance_query(q: str) -> str: return q
    def expand_query_parallel(q: str) -> list[str]: return [q]
    async def translate_query(q: str, is_only_meta_mode: bool = False) -> str: return q

# --- Cấu hình DRES và hệ thống ---
DRES_BASE_URL = "http://192.168.20.156:5601"
VIDEO_BASE_DIR = "/workspace/mlcv1/Datasets/HCMAI25/full"
IMAGE_BASE_PATH = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Retrieval/Keyframes/webp_keyframes"

BEIT3_WORKER_URL = "http://model-workers2:8001/embed"
BGE_WORKER_URL = "http://model-workers:8002/embed"
OPS_MM_WORKER_URL = "http://model-workers:8004/embed"
IMAGE_GEN_WORKER_URL = "http://localhost:8004/generate"
BGE_M3_WORKER_URL = "http://model-workers:8003/embed"
METACLIP2_WORKER_URL = "http://model-workers:8006/embed" 


ELASTICSEARCH_HOST = "http://elasticsearch2:9200"
OCR_ASR_INDEX_NAME = "batch_12_a"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"

BEIT3_COLLECTION_NAME = "beit3_batch1_2_filter"
BGE_COLLECTION_NAME = "bge_batch1_2_filter"
BGE_M3_CAPTION_COLLECTION_NAME = "BGE_M3_HCMAIC_captions_batch_1"
OPS_MM_COLLECTION_NAME = "MM_EMBED_MAX_RESOLUTION"
METACLIP2_COLLECTION_NAME = "metaclip2_from_npz"

MODEL_WEIGHTS = {"ops_mm": 0.40, "beit3": 0.25, "metaclip2": 0.20, "bge": 0.15, "bge_caption": 0.0}
SEARCH_DEPTH = 500
TOP_K_RESULTS = 1000
MAX_SEQUENCES_TO_RETURN = 500
SEARCH_DEPTH_PER_STAGE = 500
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720

# --- PERFORMANCE OPTIMIZATION: Tuned Milvus search parameters ---
SEARCH_PARAMS = {
    "HNSW": {"metric_type": "IP", "params": {"ef": 1024}}, 
    "IVF_FLAT": {"metric_type": "COSINE", "params": {"nprobe": 24}},
    "SCANN": {"metric_type": "COSINE", "params": {"nprobe": 128}},
    "DEFAULT": {"metric_type": "IP", "params": {}}
}

COLLECTION_TO_INDEX_TYPE = {
    BEIT3_COLLECTION_NAME: "HNSW",
    BGE_COLLECTION_NAME: "HNSW",
    BGE_M3_CAPTION_COLLECTION_NAME: "HNSW",
    OPS_MM_COLLECTION_NAME: "HNSW",
    METACLIP2_COLLECTION_NAME: "HNSW" 

}

es = None
OBJECT_COUNTS_DF: Optional[pl.DataFrame] = None
OBJECT_POSITIONS_DF: Optional[pl.DataFrame] = None

beit3_collection: Optional[Collection] = None
bge_collection: Optional[Collection] = None
bge_m3_caption_collection: Optional[Collection] = None
ops_mm_collection: Optional[Collection] = None
metaclip2_collection: Optional[Collection] = None


FRAME_CONTEXT_CACHE_FILE = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/DataPreprocessing/KF/frame_context_cache_4.json"
CONTEXT_IMAGE_BASE_PATH = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Retrieval/Keyframes/webp_keyframes" 
FRAME_CONTEXT_CACHE: Optional[Dict[str, List[str]]] = None

# ## TEAMWORK: Connection Manager for WebSockets ##
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
# ## END TEAMWORK ##
trake_panel_state: List[Dict[str, Any]] = [] 

# --- Pydantic Models ---
class ObjectCountFilter(BaseModel): conditions: Dict[str, str] = {}
class PositionBox(BaseModel): label: str; box: List[float]
class ObjectPositionFilter(BaseModel): boxes: List[PositionBox] = []
class ObjectFilters(BaseModel): counting: Optional[ObjectCountFilter] = None; positioning: Optional[ObjectPositionFilter] = None
class StageData(BaseModel):
    query: str
    enhance: bool
    expand: bool
    use_bge_caption: bool = False
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    query_image_name: Optional[str] = None
    generated_image_name: Optional[str] = None

class TemporalSearchRequest(BaseModel):
    stages: list[StageData]
    models: List[str] = ["beit3", "bge", "ops_mm"]
    cluster: bool = False
    filters: Optional[ObjectFilters] = None
    ambiguous: bool = False
    page: int = 1
    page_size: int = 30
    is_only_meta_mode: bool = False

class ProcessQueryRequest(BaseModel):
    query: str
    enhance: bool = False
    expand: bool = False
    is_only_meta_mode: bool = False
    models: Optional[List[str]] = None
class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_name: Optional[str] = None
    image_search_text: Optional[str] = None
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    models: List[str] = ["beit3", "bge", "ops_mm"]
    filters: Optional[ObjectFilters] = None
    enhance: bool = False
    expand: bool = False
    use_bge_caption: bool = False    
    generated_image_name: Optional[str] = None
    page: int = 1
    page_size: int = 30
    is_only_meta_mode: bool = False

class CheckFramesRequest(BaseModel): base_filepath: str
class DRESLoginRequest(BaseModel): username: str; password: str
class DRESSubmitRequest(BaseModel):
    sessionId: str
    evaluationId: str
    video_id: str
    filepath: str
    frame_id: Optional[int] = None

class VideoInfoResponse(BaseModel):
    fps: float

class TaskContentRequest(BaseModel):
    task_name: str

@app.on_event("startup")
def startup_event():
    global es, OBJECT_COUNTS_DF, OBJECT_POSITIONS_DF, beit3_collection, bge_collection, bge_m3_caption_collection, ops_mm_collection, metaclip2_collection
    try:
        print("--- Loading cache json ---")
        load_frame_context_cache_from_json()
        print("--- Loading cache json successfully ---")
    
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Milvus connection successful. ---")
        print("--- Loading Milvus collections into memory... ---")
        
        collections_to_load = {
            "BEiT3": (BEIT3_COLLECTION_NAME, "beit3"),
            "BGE": (BGE_COLLECTION_NAME, "bge"),
            "BGECaption": (BGE_M3_CAPTION_COLLECTION_NAME, "bge_caption"),
            "OpsMM": (OPS_MM_COLLECTION_NAME, "ops_mm"),
            "MetaCLIP2": (METACLIP2_COLLECTION_NAME, "metaclip2")
        }

        for name, (col_name, var_name) in collections_to_load.items():
            if utility.has_collection(col_name):
                collection = Collection(col_name)
                collection.load()
                
                if var_name == "beit3":
                    beit3_collection = collection
                elif var_name == "bge":
                    bge_collection = collection
                elif var_name == "bge_caption":
                    bge_m3_caption_collection = collection
                elif var_name == "ops_mm": 
                    ops_mm_collection = collection
                elif var_name == "metaclip2":
                    metaclip2_collection = collection
                    
                print(f"--- Collection '{col_name}' (for {name}) loaded successfully. ---")
            else:
                print(f"!!! WARNING: Collection '{col_name}' (for {name}) not found. !!!")

    except Exception as e:
        print(f"FATAL: Could not connect to or load from Milvus. Error: {e}")
        traceback.print_exc()
        
    try:
        es = Elasticsearch(ELASTICSEARCH_HOST)
        if es.ping():
            print("--- Elasticsearch connection successful. ---")
        else:
            print("FATAL: Could not connect to Elasticsearch.")
            es = None
    except Exception as e:
        print(f"FATAL: Could not connect to Elasticsearch. Error: {e}")
        es = None
        
    try:
        print("--- Loading object detection data... ---")
        counts_path = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Object/object_counts.parquet"
        positions_path = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Object/object_positions.parquet"
        counts_df = pl.read_parquet(counts_path)
        OBJECT_COUNTS_DF = counts_df.with_columns(pl.col("image_name").str.split(".").list.first().alias("name_stem"))
        positions_df = pl.read_parquet(positions_path)
        OBJECT_POSITIONS_DF = positions_df.with_columns([((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("bbox_area"), pl.col("image_name").str.split(".").list.first().alias("name_stem")])
        print(f"--- Object data loaded. ---")
    except Exception as e:
        print(f"!!! WARNING: Could not load object parquet files. Filtering disabled. Error: {e} !!!")
        OBJECT_COUNTS_DF = None
        OBJECT_POSITIONS_DF = None

# --- Helper Functions ---

def load_frame_context_cache_from_json():
    global FRAME_CONTEXT_CACHE
    if not os.path.exists(FRAME_CONTEXT_CACHE_FILE):
        print(f"!!! WARNING: Frame context cache file not found at '{FRAME_CONTEXT_CACHE_FILE}'. Context view will be disabled.")
        FRAME_CONTEXT_CACHE = {}
        return

    print(f"--- Loading Frame Context Cache from '{FRAME_CONTEXT_CACHE_FILE}'... ---")
    try:
        with open(FRAME_CONTEXT_CACHE_FILE, 'r', encoding='utf-8') as f:
            FRAME_CONTEXT_CACHE = json.load(f)
        print(f"--- Frame Context Cache loaded successfully. Cached items: {len(FRAME_CONTEXT_CACHE)} ---")
    except Exception as e:
        print(f"!!! ERROR: Could not load frame context cache. Error: {e} !!!")
        FRAME_CONTEXT_CACHE = {}

def get_video_fps(video_path: str) -> float:
    """
    Gets the FPS of a video, trying pyav first for better codec support.
    Falls back to OpenCV.
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video path does not exist for FPS check: {video_path}. Returning default.")
        return 30.0

    # Try pyav first as it's more robust with modern codecs like AV1
    if PYAV_AVAILABLE:
        try:
            container = av.open(video_path)
            fps = container.streams.video[0].average_rate
            container.close()
            if fps:
                return float(fps)
        except Exception as e:
            print(f"pyav failed to get FPS for {video_path}: {e}. Falling back to OpenCV.")

    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"cv2: Could not open video file to get FPS: {video_path}. Returning default.")
            return 30.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and fps > 0 else 30.0
    except Exception as e:
        print(f"cv2: Error getting FPS for {video_path}: {e}. Returning default.")
        return 30.0

async def get_frame_from_video(video_path: str, timestamp_sec: float) -> Optional[bytes]:
    """
    Extracts a single frame from a video at a specific timestamp.
    Intelligently uses pyav for AV1 codecs/mkv and OpenCV for others.
    Returns the frame as JPEG encoded bytes, or None if failed.
    """
    if not os.path.isfile(video_path):
        print(f"Warning: Video file not found at {video_path}")
        return None

    use_pyav = PYAV_AVAILABLE and video_path.lower().endswith(('.mkv', '.webm'))

    if use_pyav:
        try:
            container = await asyncio.to_thread(av.open, video_path)
            stream = container.streams.video[0]
            
            # Use pyav's fast and precise seeking
            target_pts = int(timestamp_sec * stream.time_base.denominator / stream.time_base.numerator)
            
            # Seeking and decoding can be blocking, run in a thread
            def _seek_and_decode():
                container.seek(target_pts, backward=True, any_frame=False, stream=stream)
                return next(container.decode(stream), None)

            frame = await asyncio.to_thread(_seek_and_decode)

            if frame is None:
                print(f"pyav: Could not decode frame at timestamp {timestamp_sec} for {video_path}")
                container.close()
                return None
                
            img = frame.to_ndarray(format='bgr24')
            
            def _encode_frame():
                return cv2.imencode(".jpg", img)

            is_success, buffer = await asyncio.to_thread(_encode_frame)

            if not is_success:
                print(f"pyav: Failed to encode frame to JPEG for {video_path}")
                container.close()
                return None
            
            container.close()
            return buffer.tobytes()
        except Exception as e:
            print(f"Error processing video with pyav: {e}. Falling back to OpenCV.")
            pass

    # Default to OpenCV for all other formats (MP4, AVI, MOV, etc.) or if pyav failed
    try:
        def _read_frame_cv2():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, None
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
            ret, frame = cap.read()
            cap.release()
            return ret, frame

        ret, frame = await asyncio.to_thread(_read_frame_cv2)
        
        if not ret or frame is None:
            print(f"cv2: Could not read frame at timestamp {timestamp_sec} for {video_path}")
            return None
            
        is_success, buffer = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
        if not is_success:
            print(f"cv2: Failed to encode frame to JPEG for {video_path}")
            return None
            
        return buffer.tobytes()
    except Exception as e:
        print(f"Error processing video with OpenCV: {e}")
        return None

def process_and_cluster_results_optimized(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    shots_by_video = defaultdict(list)
    for res in results:
        if not all(k in res for k in ['video_id', 'shot_id']):
            continue
        try:
            res['shot_id_int'] = int(str(res['shot_id']))
            shots_by_video[res['video_id']].append(res)
        except (ValueError, TypeError):
            continue
    all_clusters = []
    for video_id, shots_in_video in shots_by_video.items():
        if not shots_in_video:
            continue
        sorted_shots = sorted(shots_in_video, key=lambda x: x['shot_id_int'])
        if not sorted_shots:
            continue
        current_cluster = [sorted_shots[0]]
        for i in range(1, len(sorted_shots)):
            current_shot = sorted_shots[i]
            last_shot_in_cluster = current_cluster[-1]
            if current_shot['shot_id_int'] == last_shot_in_cluster['shot_id_int'] or current_shot['shot_id_int'] == last_shot_in_cluster['shot_id_int'] + 1:
                current_cluster.append(current_shot)
            else:
                all_clusters.append(current_cluster)
                current_cluster = [current_shot]
        if current_cluster:
            all_clusters.append(current_cluster)
    if not all_clusters:
        return []
    processed_clusters = []
    for cluster_shots in all_clusters:
        if not cluster_shots:
            continue
        sorted_cluster_shots = sorted(
            cluster_shots,
            key=lambda x: x.get('rrf_score', x.get('score', 0)),
            reverse=True
        )
        best_shot = sorted_cluster_shots[0]
        max_score = best_shot.get('rrf_score', best_shot.get('score', 0))
        processed_clusters.append({
            "cluster_score": max_score,
            "shots": sorted_cluster_shots,
            "best_shot": best_shot
        })
    return sorted(processed_clusters, key=lambda x: x['cluster_score'], reverse=True)

def get_filename_stem(filepath: str) -> Optional[str]:
    if not filepath: return None
    try: return os.path.splitext(os.path.basename(filepath))[0]
    except Exception: return None

def is_temporal_sequence_valid(sequence: Dict, filters: ObjectFilters) -> bool:
    checklist = set()
    if filters.counting and filters.counting.conditions:
        for obj, cond in filters.counting.conditions.items():
            checklist.add(f"count_{obj}_{cond}")
    if filters.positioning and filters.positioning.boxes:
        for i, pbox in enumerate(filters.positioning.boxes):
            checklist.add(f"pos_{i}_{pbox.label}")
    if not checklist:
        return True

    sequence_filepaths = {s['filepath'] for s in sequence.get('shots', []) if 'filepath' in s}
    for cluster in sequence.get('clusters', []):
        for shot in cluster.get('shots', []):
            if 'filepath' in shot:
                sequence_filepaths.add(shot['filepath'])
    if not sequence_filepaths:
        return False
    
    sequence_stems = {get_filename_stem(p) for p in sequence_filepaths if p}

    for stem in sequence_stems:
        if not checklist:
            break
        
        if filters.counting and OBJECT_COUNTS_DF is not None:
            frame_counts = OBJECT_COUNTS_DF.filter(pl.col("name_stem") == stem)
            if not frame_counts.is_empty():
                for obj, cond_str in filters.counting.conditions.items():
                    key = f"count_{obj}_{cond_str}"
                    if key in checklist:
                        op, val = parse_condition(cond_str)
                        if op and val is not None and obj in frame_counts.columns and op(frame_counts.row(0, named=True)[obj], val):
                            checklist.remove(key)

        if filters.positioning and OBJECT_POSITIONS_DF is not None:
            frame_positions = OBJECT_POSITIONS_DF.filter(pl.col("name_stem") == stem)
            if not frame_positions.is_empty():
                for i, p_box in enumerate(filters.positioning.boxes):
                    key = f"pos_{i}_{p_box.label}"
                    if key in checklist and p_box.label in frame_positions['object']:
                        intersect_area = (pl.min_horizontal([pl.col("x_max"), pl.lit(p_box.box[2] * IMAGE_WIDTH)]) - pl.max_horizontal([pl.col("x_min"), pl.lit(p_box.box[0] * IMAGE_WIDTH)])).clip(lower_bound=0) * \
                                       (pl.min_horizontal([pl.col("y_max"), pl.lit(p_box.box[3] * IMAGE_HEIGHT)]) - pl.max_horizontal([pl.col("y_min"), pl.lit(p_box.box[1] * IMAGE_HEIGHT)])).clip(lower_bound=0)

                        match_df = frame_positions.filter(pl.col("object") == p_box.label).with_columns(
                            overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)
                        ).filter(pl.col("overlap_ratio") >= 0.75)
                        
                        if not match_df.is_empty():
                            checklist.remove(key)

    return not checklist

def parse_condition(condition_str: str) -> tuple[Any, int]:
    try: return operator.ge, int(condition_str)
    except ValueError:
        op_map = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq, "=": operator.eq}
        for op_str in [">=", "<=", "==", ">", "<", "="]:
            if condition_str.startswith(op_str):
                try: return op_map[op_str], int(condition_str[len(op_str):])
                except (ValueError, TypeError): return None, None
    return None, None

def get_valid_filepaths_for_strict_search(all_filepaths: set, filters: ObjectFilters) -> set:
    candidate_stems = {get_filename_stem(p) for p in all_filepaths if p}
    if not candidate_stems: return set()

    valid_stems_df = pl.DataFrame({"name_stem": list(candidate_stems)})

    if filters.counting and OBJECT_COUNTS_DF is not None and filters.counting.conditions:
        expressions = []
        for obj, cond_str in filters.counting.conditions.items():
            op, val = parse_condition(cond_str)
            if op and val is not None and obj in OBJECT_COUNTS_DF.columns:
                expressions.append(op(pl.col(obj), val))
        
        if expressions:
            count_matches_df = OBJECT_COUNTS_DF.lazy().join(
                valid_stems_df.lazy(), on="name_stem", how="inner"
            ).filter(
                pl.all_horizontal(expressions)
            ).select("name_stem").collect()
            valid_stems_df = count_matches_df

    if filters.positioning and OBJECT_POSITIONS_DF is not None and filters.positioning.boxes:
        if valid_stems_df.is_empty(): return set()

        positions_subset_df = OBJECT_POSITIONS_DF.lazy().join(
            valid_stems_df.lazy(), on="name_stem", how="inner"
        ).collect()

        stems_satisfying_all_boxes = set(valid_stems_df["name_stem"])

        for p_box in filters.positioning.boxes:
            user_x_min_lit, user_y_min_lit, user_x_max_lit, user_y_max_lit = [pl.lit(v) for v in [p_box.box[0] * IMAGE_WIDTH, p_box.box[1] * IMAGE_HEIGHT, p_box.box[2] * IMAGE_WIDTH, p_box.box[3] * IMAGE_HEIGHT]]

            intersect_area = (pl.min_horizontal([pl.col("x_max"), user_x_max_lit]) - pl.max_horizontal([pl.col("x_min"), user_x_min_lit])).clip(lower_bound=0) * \
                           (pl.min_horizontal([pl.col("y_max"), user_y_max_lit]) - pl.max_horizontal([pl.col("y_min"), user_y_min_lit])).clip(lower_bound=0)

            stems_with_match_for_this_box = positions_subset_df.filter(
                pl.col("object") == p_box.label
            ).with_columns(
                overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)
            ).filter(
                pl.col("overlap_ratio") >= 0.75
            ).select("name_stem").unique()["name_stem"].to_list()
            
            stems_satisfying_all_boxes.intersection_update(stems_with_match_for_this_box)

        valid_stems_df = pl.DataFrame({"name_stem": list(stems_satisfying_all_boxes)})

    final_valid_stems = set(valid_stems_df["name_stem"])
    return {fp for fp in all_filepaths if get_filename_stem(fp) in final_valid_stems}


def search_milvus_sync(collection: Collection, collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        if not collection or not query_vectors: return []
        
        need_load = False
        try:
            state = utility.load_state(collection.name)
            if _MilvusLoadState and isinstance(state, _MilvusLoadState): need_load = (state != _MilvusLoadState.Loaded)
            else: need_load = (str(getattr(state, "name", str(state))).lower() != "loaded" and str(state) != "2")
        except Exception: need_load = True
        
        if need_load:
            collection.load()

        index_type = COLLECTION_TO_INDEX_TYPE.get(collection_name, "HNSW")
        search_params = SEARCH_PARAMS.get(index_type, SEARCH_PARAMS["HNSW"])
        
        results = collection.search(
            data=query_vectors,
            anns_field="vector_embedding",
            param=search_params,
            limit=limit,
            output_fields=["frame_name", "video_id", "shot_id", "frame_id"],
            expr=expr
        )
        
        final_results = []
        for one_query_hits in results:
            for hit in one_query_hits:
                entity = hit.entity
                frame_name = entity.get("frame_name")
                if not frame_name: continue
                if frame_name.endswith(".webp"): frame_name = frame_name[:-5]
                    
                frame_name = frame_name + ".webp"
                
                final_results.append({
                    "frame_name": frame_name, 
                    "score": hit.distance,
                    "video_id": entity.get("video_id"),
                    "frame_id": entity.get("frame_id"),
                    "shot_id": str(entity.get("shot_id"))
                })
                
        return final_results
        
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}")
        traceback.print_exc()
        return []

def search_ocr_on_elasticsearch_sync(keyword: str, limit: int = 500):
    if not es: return []
    query = {"_source": ["file_path", "video_id", "shot_id", "frame_id"], "query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit['_source']
            if all(k in source for k in ['file_path', 'video_id', 'shot_id', 'frame_id']):
                # Lấy tên file từ đường dẫn đầy đủ
                frame_name = os.path.basename(source['file_path'])
                results.append({
                    "frame_name": frame_name, # <--- TRẢ VỀ 'frame_name'
                    "score": hit['_score'], 
                    "video_id": source['video_id'], 
                    "shot_id": str(source['shot_id']), 
                    "frame_id": source['frame_id']
                })
        return results
    except NotFoundError: return []
    except Exception as e: print(f"Lỗi Elasticsearch OCR: {e}"); return []

def search_asr_on_elasticsearch_sync(keyword: str, limit: int = 500):
    if not es: return []
    query = {"_source": ["file_path", "video_id", "shot_id", "frame_id"], "query": {"multi_match": {"query": keyword, "fields": ["asr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit['_source']
            if all(k in source for k in ['file_path', 'video_id', 'shot_id', 'frame_id']):
                frame_name = os.path.basename(source['file_path'])
                results.append({
                    "frame_name": frame_name, 
                    "score": hit['_score'], 
                    "video_id": source['video_id'], 
                    "shot_id": str(source['shot_id']), 
                    "frame_id": source['frame_id']
                })
        return results  
    except NotFoundError: return []
    except Exception as e: print(f"Lỗi Elasticsearch ASR: {e}"); return []

async def search_milvus_async(collection: Collection, collection_name: str, query_vectors: list, limit: int, expr: str = None):
    return await asyncio.to_thread(search_milvus_sync, collection, collection_name, query_vectors, limit, expr)

async def search_ocr_on_elasticsearch_async(keyword: str, limit: int = 500):
    return await asyncio.to_thread(search_ocr_on_elasticsearch_sync, keyword, limit)

async def search_asr_on_elasticsearch_async(keyword: str, limit: int = 500):
    return await asyncio.to_thread(search_asr_on_elasticsearch_sync, keyword, limit)

def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    master_data = defaultdict(lambda: {"raw_scores": {}})
    for model_name, results in results_lists.items():
        if not results: continue
        sorted_results = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            # === SỬA Ở ĐÂY: DÙNG frame_name LÀM KHÓA CHÍNH ===
            frame_name = result.get('frame_name')
            if not frame_name: continue
            
            if 'metadata' not in master_data[frame_name]:
                master_data[frame_name]['metadata'] = result
            master_data[frame_name]['raw_scores'][model_name] = {"score": result.get('score', 0.0), "rank": rank}
            
    if not master_data: return []

    final_results = []
    # === SỬA Ở ĐÂY: Lặp qua frame_name ===
    for frame_name, data in master_data.items():
        rrf_score = 0.0
        for model_name, score_info in data['raw_scores'].items():
            model_weight = weights.get(model_name, 1.0)
            rrf_score += model_weight * (1.0 / (k_rrf + score_info['rank']))
        final_item = data['metadata']
        final_item['rrf_score'] = rrf_score
        final_item['source_scores'] = data['raw_scores'] 
        final_item.pop('score', None)
        final_results.append(final_item)
        
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)

async def get_embeddings_for_query(
    client: httpx.AsyncClient,
    text_queries: List[str],
    image_content: Optional[bytes],
    models: List[str],
    query_image_info: Optional[Dict] = None,
    use_bge_caption: bool = False
) -> Dict[str, List[List[float]]]:
    
    models_to_call = list(models)
    if use_bge_caption and "bge-m3" not in models_to_call:
        models_to_call.append("bge-m3")

    global embedding_cache
    if len(embedding_cache) > CACHE_MAX_SIZE: embedding_cache.clear()

    cache_key = f"{'|'.join(sorted(models_to_call))}:{'|'.join(text_queries or [])}"
    if image_content: pass
    elif cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    results = await get_embeddings_for_query_from_worker(
        client, text_queries, image_content, models_to_call, query_image_info
    )
    
    embedding_cache[cache_key] = results
    return results

async def get_embeddings_for_query_from_worker(
    client: httpx.AsyncClient,
    text_queries: List[str],
    image_content: Optional[bytes],
    models: List[str],
    query_image_info: Optional[Dict] = None,
    is_fusion: bool = False
) -> Dict[str, List[List[float]]]:
    tasks = []
    model_url_map = {"beit3": BEIT3_WORKER_URL, "bge": BGE_WORKER_URL, "ops_mm": OPS_MM_WORKER_URL, "bge-m3": BGE_M3_WORKER_URL, "metaclip2": METACLIP2_WORKER_URL}
    async def get_model_embedding(model_name: str) -> tuple[str, list]:
        url = model_url_map.get(model_name)
        if not url: return model_name, []
        try:
            embeddings = []
            queries = text_queries or [""]
            for q in queries:
                data = {'text_query': q} if q else {}
                files = None
                if image_content:
                    files = {'image_file': (query_image_info['filename'], image_content, query_image_info['content_type'])}
                resp = await client.post(url, files=files, data=data, timeout=20.0)
                if resp.status_code == 200:
                    embeddings.extend(resp.json().get('embedding', []))
            return model_name, embeddings
        except Exception as e:
            print(f"Error getting embedding for {model_name}: {e}")
            return model_name, []
    for model in models:
        if model in model_url_map:
            tasks.append(get_model_embedding(model))
    results = await asyncio.gather(*tasks)
    return {model_name: vecs for model_name, vecs in results}

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():                                                                                                                                                                      
    ui_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=500, detail="UI file (index.html) not found in 'templates' folder.")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global trake_panel_state 
    try:
        await websocket.send_text(json.dumps({"type": "trake_sync", "data": trake_panel_state}))
    except Exception as e:
        print(f"Initial sync failed for a client: {e}")

    try:
        while True:
            raw_data = await websocket.receive_text()
            message = json.loads(raw_data)
            msg_type = message.get("type")

            if msg_type in ["new_frame", "remove_frame", "clear_panel"]:
                await manager.broadcast(raw_data)
            
            elif msg_type == "trake_add":
                shot_data = message.get("data", {}).get("shot")
                if shot_data and not any(item['filepath'] == shot_data.get('filepath') for item in trake_panel_state):
                    trake_panel_state.append(shot_data)
                    await manager.broadcast(json.dumps({"type": "trake_add", "data": {"shot": shot_data}}))

            elif msg_type == "trake_remove":
                filepath = message.get("data", {}).get("filepath")
                if filepath:
                    trake_panel_state = [item for item in trake_panel_state if item.get('filepath') != filepath]
                    await manager.broadcast(raw_data)

            elif msg_type == "trake_reorder":
                new_order_filepaths = message.get("data", {}).get("order")
                if isinstance(new_order_filepaths, list):
                    current_items_map = {item['filepath']: item for item in trake_panel_state}
                    new_state = [current_items_map[fp] for fp in new_order_filepaths if fp in current_items_map]
                    trake_panel_state = new_state
                    await manager.broadcast(raw_data)
            
            elif msg_type == "trake_replace":
                data = message.get("data", {})
                filepath_to_replace = data.get("filepath")
                new_shot_data = data.get("newShot")
                if filepath_to_replace and new_shot_data:
                    for i, item in enumerate(trake_panel_state):
                        if item.get("filepath") == filepath_to_replace:
                            trake_panel_state[i] = new_shot_data
                            break
                    await manager.broadcast(raw_data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        traceback.print_exc()
        manager.disconnect(websocket)

@app.post("/process_query")
async def process_query(request_data: ProcessQueryRequest):
    query = request_data.query
    if not query:
        return {"processed_query": ""}

    if len(processed_query_cache) > CACHE_MAX_SIZE:
        processed_query_cache.clear()
        
    models_key = '|'.join(sorted(request_data.models)) if request_data.models else ''
    cache_key = f"{query}|{request_data.enhance}|{request_data.expand}|{request_data.is_only_meta_mode}|{models_key}"
    if cache_key in processed_query_cache:
        return {"processed_query": processed_query_cache[cache_key]}

    # Determine if translation should be skipped
    effective_meta_mode = request_data.is_only_meta_mode
    if request_data.models == ["metaclip2"]:
        effective_meta_mode = True

    processed_query = ""
    if request_data.enhance:
        processed_query = await asyncio.to_thread(enhance_query, query)
    elif request_data.expand:
        translated_query = await translate_query(query, is_only_meta_mode=effective_meta_mode)
        processed_query = await asyncio.to_thread(expand_query_parallel, translated_query)
    else:
        processed_query = await translate_query(query, is_only_meta_mode=effective_meta_mode)
        
    if isinstance(processed_query, list):
        processed_query = "\n".join(processed_query)
        
    processed_query_cache[cache_key] = processed_query
    return {"processed_query": processed_query}

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    extension = Path(image.filename).suffix
    temp_filename = f"{uuid.uuid4()}{extension}"
    temp_filepath = TEMP_UPLOAD_DIR / temp_filename
    try:
        with temp_filepath.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    finally:
        image.file.close()
    return {"temp_image_name": temp_filename}

@app.post("/dres/login")
async def dres_login(login_data: DRESLoginRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{DRES_BASE_URL}/api/v2/login", json=login_data.dict())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"DRES login failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")

@app.get("/dres/list_evaluations")
async def dres_list_evaluations(session: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{DRES_BASE_URL}/api/v2/client/evaluation/list", params={"session": session})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"DRES list evaluations failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")
def url_safe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')

@app.post("/dres/submit")
async def dres_submit(submit_data: DRESSubmitRequest):
    try:
        if submit_data.frame_id is None:
            raise ValueError("frame_id is missing.")

        video_filename = submit_data.video_id
        if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_filename += '.mp4'
        
        video_full_path = os.path.join(VIDEO_BASE_DIR, video_filename)
        video_fps = await asyncio.to_thread(get_video_fps, video_full_path)
        time_ms = int((submit_data.frame_id / video_fps) * 1000)
        video_item_name = os.path.splitext(submit_data.video_id)[0]
        
        submission_body = {
            "answerSets": [{
                "answers": [{
                    "mediaItemName": video_item_name,
                    "start": time_ms,
                    "end": time_ms
                }]
            }]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DRES_BASE_URL}/api/v2/submit/{submit_data.evaluationId}",
                params={"session": submit_data.sessionId},
                json=submission_body,
                timeout=30.0 # Good practice to add a timeout
            )
            response.raise_for_status()
            dres_result = response.json()
            
            # --- ENHANCED: WebSocket Broadcasting Logic ---
            submission_status = dres_result.get("submission")
            
            if submission_status == "CORRECT":
                # Create the shotData object for the UI
                shot_data_for_ui = {
                    "video_id": submit_data.video_id,
                    "frame_id": submit_data.frame_id,
                    "filepath": submit_data.filepath,
                    # We need the URL of the image to display it
                    "url": f"/images/{url_safe_b64encode(submit_data.filepath.encode('utf-8'))}"
                }
                
                # Broadcast the global "correct" message to all clients
                await manager.broadcast(json.dumps({
                    "type": "global_correct_submission",
                    "data": {
                        "shot": shot_data_for_ui
                    }
                }))
                
                await manager.broadcast(json.dumps({"type": "clear_panel", "status": "success"}))
            
            elif submission_status == "WRONG":
                # Broadcast the "wrong" status update to all clients
                await manager.broadcast(json.dumps({
                    "type": "submission_status_update",
                    "data": {
                        "filepath": submit_data.filepath,
                        "status": "WRONG"
                    }
                }))
            elif submission_status == "DUPLICATE":
                 await manager.broadcast(json.dumps({
                    "type": "submission_status_update",
                    "data": {
                        "filepath": submit_data.filepath,
                        "status": "DUPLICATE"
                    }
                }))
            # --- END: New Logic ---

            return dres_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"DRES submission failed: {e.response.text}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    
class ImageGenTextRequest(BaseModel):
    query: str
    enhance: bool = False
    expand: bool = False

@app.post("/generate_image_from_text")
async def generate_image_from_text(request_data: ImageGenTextRequest):
    if not request_data.query: raise HTTPException(status_code=400, detail="Query text is required.")
    processed_text = await translate_query(request_data.query)
    if request_data.expand: processed_text = " ".join(await asyncio.to_thread(expand_query_parallel, processed_text))
    if request_data.enhance: processed_text = await asyncio.to_thread(enhance_query, processed_text)
    try:
        response = requests.post(IMAGE_GEN_WORKER_URL, json={"query": processed_text}, timeout=60.0)
        response.raise_for_status()
        temp_filename = f"gen_{uuid.uuid4()}.png"
        temp_filepath = TEMP_UPLOAD_DIR / temp_filename
        with temp_filepath.open("wb") as f: f.write(response.content)
        encoded_path = base64.urlsafe_b64encode(str(temp_filepath).encode('utf-8')).decode('utf-8')
        image_url = f"/images/{encoded_path}"
        return {"temp_image_name": temp_filename, "image_url": image_url}
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=502, detail=f"Failed to contact image generation service: {e}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error during image generation process: {e}")

# --- Search Endpoints ---
@app.post("/search")
async def search_unified(request: Request, search_data: str = Form(...), query_image: Optional[UploadFile] = File(None)):
    start_total_time = time.time()
    timings = {}
    try:
        search_data_model = UnifiedSearchRequest.parse_raw(search_data)
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid search data format: {e}")

    # <--- START CORRECTION: BATCHING LOGIC FOR /search ENDPOINT --->
    
    final_fused_results = []
    processed_query_for_ui = ""

    start_query_proc = time.time()
    final_queries_to_embed, image_content, query_image_info = [], None, None
    models_to_use = search_data_model.models
    is_image_search = bool(query_image or search_data_model.query_image_name)
    is_gen_image_search = bool(search_data_model.generated_image_name)

    if is_gen_image_search:
        models_to_use = ["unite"]
        base_query = await translate_query(search_data_model.query_text)
        final_queries_to_embed = [base_query]
        temp_filepath = TEMP_UPLOAD_DIR / search_data_model.generated_image_name
        if temp_filepath.is_file():
            image_content = temp_filepath.read_bytes()
            query_image_info = {"filename": search_data_model.generated_image_name, "content_type": "image/png"}
    elif is_image_search:
        models_to_use = ["bge"]
        if search_data_model.image_search_text:
            final_queries_to_embed = [await translate_query(search_data_model.image_search_text)]
        if query_image:
            image_content, query_image_info = await query_image.read(), {"filename": query_image.filename, "content_type": query_image.content_type}
        elif search_data_model.query_image_name:
            temp_filepath = TEMP_UPLOAD_DIR / search_data_model.query_image_name
            if temp_filepath.is_file():
                image_content, query_image_info = temp_filepath.read_bytes(), {"filename": search_data_model.query_image_name, "content_type": "image/jpeg"}
    elif search_data_model.query_text:
        processed_query_for_ui = (await process_query(ProcessQueryRequest(
            query=search_data_model.query_text,
            enhance=search_data_model.enhance,
            expand=search_data_model.expand,
            is_only_meta_mode=search_data_model.is_only_meta_mode,
            models=search_data_model.models
        )))["processed_query"]
        final_queries_to_embed = [processed_query_for_ui]
    timings["query_processing_s"] = time.time() - start_query_proc

    is_primary_search = bool(final_queries_to_embed or image_content)
    is_filter_search = bool(search_data_model.ocr_query or search_data_model.asr_query)

    if not is_primary_search and not is_filter_search:
        return ORJSONResponse(content={"results": [], "processed_query": "", "total_results": 0, "timing_info": {}})

    es_results_for_standalone_search = []
    frame_batches = []
    
    if is_filter_search:
        start_es = time.time()
        es_tasks = []
        if search_data_model.ocr_query:
            es_tasks.append(search_ocr_on_elasticsearch_async(search_data_model.ocr_query, limit=500))
        if search_data_model.asr_query:
            es_tasks.append(search_asr_on_elasticsearch_async(search_data_model.asr_query, limit=500))
        
        es_results_lists = await asyncio.gather(*es_tasks)
        es_res_map = {res['filepath']: res for res_list in es_results_lists for res in res_list}
        es_results_for_standalone_search = list(es_res_map.values())
        timings["ocr_asr_filtering_s"] = time.time() - start_es

        if not es_results_for_standalone_search:
            # If filtering returns nothing, the intersection is empty. Return early.
            return ORJSONResponse(content={"results": [], "processed_query": processed_query_for_ui, "total_results": 0, "timing_info": timings})

        candidate_frame_names = [os.path.splitext(os.path.basename(res['filepath']))[0] for res in es_results_for_standalone_search]
        
        if candidate_frame_names:
            batch_size = 400
            frame_batches = [candidate_frame_names[i:i + batch_size] for i in range(0, len(candidate_frame_names), batch_size)]

    if is_primary_search:
        if is_filter_search and not frame_batches:
            # This case happens if ES search returns results but frame name extraction fails.
            return ORJSONResponse(content={"results": [], "processed_query": processed_query_for_ui, "total_results": 0, "timing_info": timings})
        
        start_embed = time.time()
        async with httpx.AsyncClient() as client:
            results_by_model = await get_embeddings_for_query(client, final_queries_to_embed, image_content, models_to_use, query_image_info, use_bge_caption=search_data_model.use_bge_caption)
        timings["embedding_generation_s"] = time.time() - start_embed
        
        if any(results_by_model.values()):
            start_milvus = time.time()

            milvus_expr = None
            # If a filter search was performed, create a single, efficient expression for Milvus
            if is_filter_search and frame_batches:
                all_candidate_names = [name for batch in frame_batches for name in batch]
                if all_candidate_names:
                    formatted_names = ", ".join([f'"{name}"' for name in all_candidate_names])
                    milvus_expr = f"frame_name in [{formatted_names}]"

            milvus_tasks = []
            models_in_task_order = []

            model_map = {
                "beit3": (beit3_collection, BEIT3_COLLECTION_NAME),
                "bge": (bge_collection, BGE_COLLECTION_NAME),
                "bge_caption": (bge_m3_caption_collection, BGE_M3_CAPTION_COLLECTION_NAME),
                "ops_mm": (ops_mm_collection, OPS_MM_COLLECTION_NAME),
                "metaclip2": (metaclip2_collection, METACLIP2_COLLECTION_NAME)
            }

            models_to_query = {m: v for m, v in results_by_model.items() if m.replace("-m3", "_caption") in model_map}

            for model_key, vectors in models_to_query.items():
                rrf_model_name = model_key.replace("-m3", "_caption")
                if rrf_model_name in model_map:
                    collection, col_name = model_map[rrf_model_name]
                    # Add one efficient search task per model to the list
                    milvus_tasks.append(search_milvus_async(collection, col_name, vectors, SEARCH_DEPTH, expr=milvus_expr))
                    models_in_task_order.append(rrf_model_name)

            # Run all model searches in parallel
            milvus_results_list = await asyncio.gather(*milvus_tasks)

            # Map the results back to the correct model names for RRF
            results_for_rrf = {model_name: result for model_name, result in zip(models_in_task_order, milvus_results_list)}

            timings["vector_search_s"] = time.time() - start_milvus
            
            start_post_proc = time.time()
            milvus_weights = {m: w for m, w in MODEL_WEIGHTS.items() if m in models_to_use or (m == "bge_caption" and search_data_model.use_bge_caption)}
            final_fused_results = reciprocal_rank_fusion(results_for_rrf, milvus_weights)
            timings["post_processing_s"] = time.time() - start_post_proc
            
    elif is_filter_search: # Only filter search was performed
        start_post_proc = time.time()
        for res in es_results_for_standalone_search:
            res['rrf_score'] = res.pop('score', 0.0)
        final_fused_results = sorted(es_results_for_standalone_search, key=lambda x: x.get('rrf_score', 0), reverse=True)
        timings["post_processing_s"] = time.time() - start_post_proc

    # <--- END CORRECTION --->

    start_final_proc = time.time()
    if final_fused_results:
        # Deduplicate results before clustering
        final_fused_results = list({res['frame_name']: res for res in final_fused_results}.values())

        clustered_results = process_and_cluster_results_optimized(final_fused_results)
        if search_data_model.filters and clustered_results:
            all_filepaths = {s['filepath'] for c in clustered_results for s in c.get('shots', []) if 'filepath' in s}
            valid_filepaths = await asyncio.to_thread(get_valid_filepaths_for_strict_search, all_filepaths, search_data_model.filters)
            final_results_all = []
            for cluster in clustered_results:
                valid_shots = [s for s in cluster.get('shots', []) if s.get('filepath') in valid_filepaths]
                if valid_shots:
                    new_cluster = cluster.copy()
                    new_cluster['shots'] = valid_shots
                    if new_cluster.get('best_shot') and new_cluster['best_shot'].get('filepath') not in valid_filepaths:
                        new_cluster['best_shot'] = max(valid_shots, key=lambda x: x.get('rrf_score', 0))
                    final_results_all.append(new_cluster)
        else:
            final_results_all = clustered_results
    else:
        final_results_all = []
    
    timings["final_processing_s"] = time.time() - start_final_proc

    total_results = len(final_results_all)
    start_index = (search_data_model.page - 1) * search_data_model.page_size
    paginated_results = final_results_all[start_index:start_index + search_data_model.page_size]

    response_content = {
        "results": paginated_results,
        "processed_query": processed_query_for_ui,
        "total_results": total_results
    }
    timings["total_request_s"] = time.time() - start_total_time
    response_content["timing_info"] = timings
    return ORJSONResponse(content=response_content)

async def _perform_es_search(ocr_query: Optional[str], asr_query: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """Helper: Thực hiện tìm kiếm trên Elasticsearch và trả về kết quả đã được chuẩn hóa."""
    if not ocr_query and not asr_query:
        return []
    
    es_tasks = []
    if ocr_query:
        es_tasks.append(search_ocr_on_elasticsearch_async(ocr_query, limit=limit))
    if asr_query:
        es_tasks.append(search_asr_on_elasticsearch_async(asr_query, limit=limit))
    
    es_results_lists = await asyncio.gather(*es_tasks)
    
    # Gộp kết quả và loại bỏ trùng lặp, giữ lại kết quả có điểm cao hơn
    es_res_map = {}
    for res_list in es_results_lists:
        for res in res_list:
            frame_name = res.get('frame_name') # Dùng frame_name
            if not frame_name: continue
            if frame_name not in es_res_map or res.get('score', 0) > es_res_map[frame_name].get('score', 0):
                es_res_map[frame_name] = res
    
    return list(es_res_map.values())

async def _perform_vector_search(
    client: httpx.AsyncClient, 
    stage: StageData, 
    requested_models: List[str],
    processed_queries_for_ui: List[str],
    is_only_meta_mode: bool = False
) -> List[Dict[str, Any]]:
    """Helper: Thực hiện toàn bộ quy trình vector search (embedding -> milvus -> rrf)."""
    results_by_model = {}
    image_content, query_image_info = None, None

    # 1. Lấy embedding
    if stage.generated_image_name and stage.query:
        # ... logic xử lý ảnh tạo ra (giữ nguyên)
        processed_queries_for_ui.append(f"Gen-Image Fusion: {stage.query}")
        temp_filepath = TEMP_UPLOAD_DIR / stage.generated_image_name
        if not temp_filepath.is_file(): return []
        image_content = temp_filepath.read_bytes()
        query_image_info = {"filename": stage.generated_image_name, "content_type": "image/png"}
        results_by_model = await get_embeddings_for_query(client, [await translate_query(stage.query)], image_content, ["unite"], query_image_info)
    elif stage.query_image_name:
        # ... logic xử lý ảnh tải lên (giữ nguyên)
        processed_queries_for_ui.append(f"Image: {stage.query_image_name}")
        temp_filepath = TEMP_UPLOAD_DIR / stage.query_image_name
        if not temp_filepath.is_file(): return []
        image_content = temp_filepath.read_bytes()
        query_image_info = {"filename": stage.query_image_name, "content_type": "image/jpeg"}
        results_by_model = await get_embeddings_for_query(client, [], image_content, ["bge"], query_image_info)
    else:
        # ... logic xử lý truy vấn văn bản (giữ nguyên)
        processed_query = (await process_query(ProcessQueryRequest(
            query=stage.query,
            enhance=stage.enhance,
            expand=stage.expand,
            is_only_meta_mode=is_only_meta_mode,
            models=requested_models
        )))["processed_query"]
        processed_queries_for_ui.append(processed_query)
        # Chỉ lấy embedding cho các model được yêu cầu
        results_by_model = await get_embeddings_for_query(client, [processed_query], None, requested_models, use_bge_caption=stage.use_bge_caption)
    
    if not any(results_by_model.values()):
        return []

    # 2. Tìm kiếm trên Milvus (luôn là global search, không có expr)
    milvus_tasks = []
    models_in_task_order = []
    model_map = {
        "beit3": (beit3_collection, BEIT3_COLLECTION_NAME), "bge": (bge_collection, BGE_COLLECTION_NAME),
        "bge_caption": (bge_m3_caption_collection, BGE_M3_CAPTION_COLLECTION_NAME), "ops_mm": (ops_mm_collection, OPS_MM_COLLECTION_NAME),
        "metaclip2": (metaclip2_collection, METACLIP2_COLLECTION_NAME)
    }
    
    # --- LOGIC IF/ELSE TƯỜNG MINH HƠN Ở ĐÂY ---
    if "beit3" in requested_models and "beit3" in results_by_model:
        collection, col_name = model_map["beit3"]
        milvus_tasks.append(search_milvus_async(collection, col_name, results_by_model["beit3"], SEARCH_DEPTH_PER_STAGE, expr=None))
        models_in_task_order.append("beit3")

    if "bge" in requested_models and "bge" in results_by_model:
        collection, col_name = model_map["bge"]
        milvus_tasks.append(search_milvus_async(collection, col_name, results_by_model["bge"], SEARCH_DEPTH_PER_STAGE, expr=None))
        models_in_task_order.append("bge")

    if "ops_mm" in requested_models and "ops_mm" in results_by_model:
        collection, col_name = model_map["ops_mm"]
        milvus_tasks.append(search_milvus_async(collection, col_name, results_by_model["ops_mm"], SEARCH_DEPTH_PER_STAGE, expr=None))
        models_in_task_order.append("ops_mm")

    if "metaclip2" in requested_models and "metaclip2" in results_by_model:
        collection, col_name = model_map["metaclip2"]
        milvus_tasks.append(search_milvus_async(collection, col_name, results_by_model["metaclip2"], SEARCH_DEPTH_PER_STAGE, expr=None))
        models_in_task_order.append("metaclip2")
        
    if stage.use_bge_caption and "bge-m3" in results_by_model:
        collection, col_name = model_map["bge_caption"]
        milvus_tasks.append(search_milvus_async(collection, col_name, results_by_model["bge-m3"], SEARCH_DEPTH_PER_STAGE, expr=None))
        models_in_task_order.append("bge_caption")
    # --- KẾT THÚC LOGIC TƯỜNG MINH ---

    if not milvus_tasks:
        return []

    milvus_results_list = await asyncio.gather(*milvus_tasks)
    results_for_rrf = {model_name: result for model_name, result in zip(models_in_task_order, milvus_results_list)}
    
    # 3. Trộn kết quả và trả về
    return reciprocal_rank_fusion(results_for_rrf, MODEL_WEIGHTS)

def _combine_and_rerank_results(
    vector_results: List[Dict[str, Any]], 
    es_results: List[Dict[str, Any]],
    vector_weight: float = 0.6,
    es_weight: float = 0.4
) -> List[Dict[str, Any]]:
    """Helper: Tìm giao điểm và xếp hạng lại kết quả từ hai nguồn."""
    vector_map = {res['frame_name']: res for res in vector_results}
    es_map = {res['frame_name']: res for res in es_results}
    
    intersection_framenames = set(vector_map.keys()).intersection(es_map.keys()) 
    
    if not intersection_framenames:
        return []
        
    final_results = []
    # Chuẩn hóa điểm số của ES để tương đương với điểm RRF (thường từ 0-1)
    max_es_score = max(res.get('score', 1.0) for res in es_results) or 1.0
    
    for fname in intersection_framenames:
        vector_res = vector_map[fname]
        es_res = es_map[fname]
        
        # Điểm RRF đã được chuẩn hóa, không cần làm gì thêm
        vector_score = vector_res.get('rrf_score', 0.0)
        # Chuẩn hóa điểm ES về thang điểm tương tự
        normalized_es_score = es_res.get('score', 0.0) / max_es_score
        
        combined_score = (vector_score * vector_weight) + (normalized_es_score * es_weight)
        
        # Sử dụng lại metadata từ kết quả vector search vì nó đầy đủ hơn (có source_scores)
        final_item = vector_res
        final_item['rrf_score'] = combined_score # Ghi đè điểm cũ bằng điểm kết hợp
        final_item['source_scores']['elasticsearch'] = {"score": es_res.get('score', 0.0), "normalized_score": normalized_es_score}
        final_results.append(final_item)
        
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)


@app.post("/temporal_search", response_class=ORJSONResponse)
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    start_total_time = time.time()
    timings = {}
    models, stages, filters, ambiguous = request_data.models, request_data.stages, request_data.filters, request_data.ambiguous
    if not stages or not models:
        raise HTTPException(status_code=400, detail="Stages and models are required.")
    
    processed_queries_for_ui = []

    # <--- START CORRECTION: BATCHING LOGIC FOR /temporal_search ENDPOINT --->
    async def get_stage_results(client: httpx.AsyncClient, stage: StageData):
        """
        PHIÊN BẢN TÁI CẤU TRÚC:
        Thực hiện tìm kiếm vector và ES song song, sau đó kết hợp kết quả.
        """
        has_vector_query = bool(stage.query or stage.query_image_name or stage.generated_image_name)
        has_ocr_asr_filter = bool(stage.ocr_query or stage.asr_query)

        # Nếu không có truy vấn nào, trả về rỗng
        if not has_vector_query and not has_ocr_asr_filter:
            return []

        # --- Giai đoạn 1: Tạo và chạy các task song song ---
        search_tasks = []
        vector_task = None
        es_task = None
        
        if has_vector_query:
            vector_task = asyncio.create_task(
                _perform_vector_search(client, stage, models, processed_queries_for_ui, request_data.is_only_meta_mode)
            )
            search_tasks.append(vector_task)
            
        if has_ocr_asr_filter:
            # Giới hạn số ứng viên ES để đảm bảo hiệu suất
            ES_CANDIDATE_LIMIT = 500 
            es_task = asyncio.create_task(
                _perform_es_search(stage.ocr_query, stage.asr_query, limit=ES_CANDIDATE_LIMIT)
            )
            search_tasks.append(es_task)
            
        await asyncio.gather(*search_tasks)

        # --- Giai đoạn 2: Lấy và xử lý kết quả ---
        vector_results = await vector_task if vector_task else []
        es_results = await es_task if es_task else []

        # --- Giai đoạn 3: Quyết định kết quả cuối cùng cho stage ---
        
        # Kịch bản 1: Chỉ tìm kiếm vector
        if has_vector_query and not has_ocr_asr_filter:
            return vector_results
            
        # Kịch bản 2: Chỉ tìm kiếm OCR/ASR
        if not has_vector_query and has_ocr_asr_filter:
            # Chuyển đổi score thành rrf_score để nhất quán
            for res in es_results:
                res['rrf_score'] = res.pop('score', 0.0)
            return sorted(es_results, key=lambda x: x.get('rrf_score', 0), reverse=True)
            
        # Kịch bản 3: Tìm kiếm kết hợp (phần quan trọng nhất)
        if has_vector_query and has_ocr_asr_filter:
            return _combine_and_rerank_results(vector_results, es_results)
            
        return [] # Trường hợp dự phòng
    # <--- END CORRECTION --->

    start_stages = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        stage_tasks = [get_stage_results(client, stage) for stage in request_data.stages]
        all_stage_candidates = await asyncio.gather(*stage_tasks, return_exceptions=True)
    timings["stage_candidate_gathering_s"] = time.time() - start_stages
    
    valid_stage_results = [res for res in all_stage_candidates if isinstance(res, list)]
    if len(valid_stage_results) < len(stages):
        response = []
        content = json.loads(response.body)
        content.update({"processed_queries": processed_queries_for_ui, "total_results": 0, "timing_info": {**timings, "total_request_s": time.time() - start_total_time}})
        return ORJSONResponse(content=content)
        
    clustered_results_by_stage = [process_and_cluster_results_optimized(list({item['frame_name']: item for item in res}.values())) for res in valid_stage_results]    
    start_assembly = time.time()
    for stage_clusters in clustered_results_by_stage:
        for cluster in stage_clusters:
            if cluster.get('shots'):
                shot_ids_int = [s['shot_id_int'] for s in cluster['shots'] if 'shot_id_int' in s]
                if shot_ids_int:
                    cluster['min_shot_id'], cluster['max_shot_id'] = min(shot_ids_int), max(shot_ids_int)
                    cluster['video_id'] = cluster['best_shot']['video_id']
    clusters_by_video = defaultdict(lambda: defaultdict(list))
    for i, stage_clusters in enumerate(clustered_results_by_stage):
        for cluster in stage_clusters:
            if 'video_id' in cluster: clusters_by_video[cluster['video_id']][i].append(cluster)
    
    all_valid_sequences = []
    if not ambiguous:
        for video_id, video_stages in clusters_by_video.items():
            if len(video_stages) < len(stages): continue
            def find_sequences_recursive(stage_idx: int, current_sequence: list):
                if stage_idx == len(stages):
                    all_valid_sequences.append(list(current_sequence))
                    return
                for next_cluster in video_stages.get(stage_idx, []):
                    if not current_sequence or next_cluster.get('min_shot_id', -1) > current_sequence[-1].get('max_shot_id', -1):
                        current_sequence.append(next_cluster)
                        find_sequences_recursive(stage_idx + 1, current_sequence)
                        current_sequence.pop()
            find_sequences_recursive(0, [])
    else:
        for video_id, video_stages in clusters_by_video.items():
            if len(video_stages) < len(stages): continue
            best_clusters_for_video = []
            for stage_idx in range(len(stages)):
                stage_clusters = video_stages.get(stage_idx, [])
                if not stage_clusters: best_clusters_for_video = []; break
                best_clusters_for_video.append(max(stage_clusters, key=lambda c: c.get('cluster_score', 0)))
            if best_clusters_for_video: all_valid_sequences.append(best_clusters_for_video)
    timings["sequence_assembly_s"] = time.time() - start_assembly
    
    if not all_valid_sequences:
        response = []
        content = json.loads(response.body)
        content.update({"processed_queries": processed_queries_for_ui, "total_results": 0, "timing_info": {**timings, "total_request_s": time.time() - start_total_time}})
        return ORJSONResponse(content=content)

    start_final_proc, TEMPORAL_PENALTY_WEIGHT = time.time(), 0.05
    processed_sequences = []
    for cluster_seq in all_valid_sequences:
        if not cluster_seq: continue
        avg_score = sum(c.get('cluster_score', 0) for c in cluster_seq) / len(cluster_seq)
        total_temporal_gap = 0
        if len(cluster_seq) > 1 and not ambiguous:
            for i in range(len(cluster_seq) - 1):
                gap = cluster_seq[i+1].get('min_shot_id', 0) - cluster_seq[i].get('max_shot_id', 0)
                if gap > 0: total_temporal_gap += gap
        combined_score = avg_score / (1 + (total_temporal_gap * TEMPORAL_PENALTY_WEIGHT))
        shots_to_display = [c['best_shot'] for c in cluster_seq] if not ambiguous else [shot for c in cluster_seq for shot in c.get('shots', [])]
        processed_sequences.append({"combined_score": combined_score, "average_rrf_score": avg_score, "temporal_gap": total_temporal_gap, "clusters": cluster_seq, "shots": shots_to_display, "video_id": cluster_seq[0].get('video_id', 'N/A')})

    sequences_to_filter = sorted(processed_sequences, key=lambda x: x['combined_score'], reverse=True)

    if filters and (filters.counting or filters.positioning):
        final_sequences_all = [seq for seq in sequences_to_filter if is_temporal_sequence_valid(seq, filters)]
    else: final_sequences_all = sequences_to_filter
    
    total_sequences = len(final_sequences_all)
    paginated_sequences = final_sequences_all[(request_data.page - 1) * request_data.page_size : request_data.page * request_data.page_size]
    timings["final_processing_s"] = time.time() - start_final_proc
    
    content = {
        "results": paginated_sequences,
        "processed_queries": processed_queries_for_ui,
        "is_temporal_search": not ambiguous,
        "is_ambiguous_search": ambiguous,
        "total_results": total_sequences
    }
    timings["total_request_s"] = time.time() - start_total_time
    content["timing_info"] = timings
    return ORJSONResponse(content=content)

@app.post("/get_frame_at_timestamp")
async def get_frame_at_timestamp(video_id: str = Form(...), timestamp: float = Form(...)):
    if "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID.")
    
    # Intelligently find the video file path with various extensions
    base_path = os.path.join(VIDEO_BASE_DIR, video_id)
    possible_paths = [
        base_path,
        f"{base_path}.mp4",
        f"{base_path}.mkv",
        f"{base_path}.webm",
        f"{base_path}.avi",
        f"{base_path}.mov"
    ]
    
    video_path_str = None
    for path in possible_paths:
        if os.path.isfile(path):
            video_path_str = path
            break
            
    if not video_path_str:
        # Final check if the original video_id had an extension
        if not any(video_id.lower().endswith(ext) for ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']):
             raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
        elif not os.path.isfile(base_path):
             raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
        else:
             video_path_str = base_path

    # Use our new powerful helper function
    frame_bytes = await get_frame_from_video(video_path_str, timestamp)

    if frame_bytes is None:
        raise HTTPException(status_code=404, detail=f"Could not read frame at timestamp {timestamp}s from {video_id}.")
    
    img_base64 = base64.b64encode(frame_bytes).decode("utf-8")
    return {"image_data": f"data:image/jpeg;base64,{img_base64}"}

@app.post("/check_temporal_frames")
async def check_temporal_frames(request_data: CheckFramesRequest) -> List[str]:
    base_filepath = request_data.base_filepath
    if not FRAME_CONTEXT_CACHE: return [base_filepath]
    key_filename = os.path.basename(base_filepath)
    neighbor_filenames = FRAME_CONTEXT_CACHE.get(key_filename)
    if not neighbor_filenames: return [base_filepath]
    return [os.path.join(CONTEXT_IMAGE_BASE_PATH, fname) for fname in neighbor_filenames]

@app.get("/videos/{video_id}")
async def get_video(video_id: str, range: str = Header(None)):
    if "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID.")
    
    base_path = os.path.join(VIDEO_BASE_DIR, video_id)
    # Define possible paths and their corresponding MIME types
    possible_files = {
        base_path: "video/mp4", # Default if no extension
        f"{base_path}.mp4": "video/mp4",
        f"{base_path}.mkv": "video/x-matroska",
        f"{base_path}.webm": "video/webm",
        f"{base_path}.avi": "video/x-msvideo",
        f"{base_path}.mov": "video/quicktime"
    }
    
    video_path = None
    content_type = "video/mp4" # Default
    for path, mime_type in possible_files.items():
        if os.path.isfile(path):
            video_path = path
            content_type = mime_type
            break

    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video not found for: {video_id}")

    video_size = os.path.getsize(video_path)
    start, end = 0, video_size - 1
    status_code = 200
    headers = {'Content-Length': str(video_size), 'Accept-Ranges': 'bytes', 'Content-Type': content_type}
    
    if range:
        range_match = re.match(r'bytes=(\d+)-(\d*)', range)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else start + 1024 * 1024 * 2
            end = min(end, video_size - 1)
            if start >= video_size or end < start:
                raise HTTPException(status_code=416, detail="Range not satisfiable")
            status_code = 206
            headers['Content-Length'] = str(end - start + 1)
            headers['Content-Range'] = f'bytes {start}-{end}/{video_size}'
    
    async def video_iterator(start_pos, end_pos):
        with open(video_path, "rb") as video_file:
            video_file.seek(start_pos)
            bytes_to_read = end_pos - start_pos + 1
            while bytes_to_read > 0:
                chunk = video_file.read(min(8192, bytes_to_read))
                if not chunk: break
                bytes_to_read -= len(chunk)
                yield chunk

    return StreamingResponse(video_iterator(start, end), status_code=status_code, headers=headers, media_type=content_type)

@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try: original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Invalid base64 path.")
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(original_path)
    if not safe_path.startswith(safe_base) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    return FileResponse(safe_path)

@app.get("/video_info/{video_id}", response_model=VideoInfoResponse)
async def get_video_info(video_id: str):
    if "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID.")
    
    base_path = os.path.join(VIDEO_BASE_DIR, video_id)
    possible_paths = [
        base_path,
        f"{base_path}.mp4",
        f"{base_path}.mkv",
        f"{base_path}.webm",
        f"{base_path}.avi",
        f"{base_path}.mov"
    ]
    
    video_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            video_path = path
            break

    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video info not found for: {video_id}")
        
    fps = await asyncio.to_thread(get_video_fps, video_path)
    return VideoInfoResponse(fps=fps)