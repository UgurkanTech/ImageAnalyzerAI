#########################################################################
#                                                                       #
#                   ImageAnalyzerAI by UgurkanTech                      #
#                                                                       #
#             Read the license information from the repository          #
#                                 2025                                  #
#########################################################################
import sys
import os
import json
import sqlite3
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time    
import ctypes

import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar, QComboBox,
    QFileDialog, QListWidget, QSplitter, QGroupBox, QSpinBox,
    QListWidgetItem, QScrollArea, QGridLayout, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QMimeData, QByteArray, QUrl, QBuffer, QIODevice, QObject
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon
import numpy as np

THREAD_COUNT = 24

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0]))) #Fix for direct curring CWD

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def get_models(self) -> List[str]:
        """Get available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
        except:
            pass
        return []
    
    def generate_description(self, model: str, image_path: str, prompt: str, context_size: int = 2048) -> str:
        """Generate description for an image"""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "num_ctx": context_size
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
        return ""
    
    def generate_embedding(self, model: str, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            if response.status_code == 200:
                return response.json().get("embedding", [])
        except:
            pass
        return []

class DatabaseManager:
    def __init__(self, data_dir: str = "data", ollama_client=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.ollama_client = ollama_client

    
    def get_db_name(self, vision_model: str, embedding_model: str = None) -> str:
        """Generate database name from model pair"""
        vision_safe = "".join(c for c in vision_model if c.isalnum() or c in ("-", "_"))
        if embedding_model:
            embedding_safe = "".join(c for c in embedding_model if c.isalnum() or c in ("-", "_"))
            return f"{vision_safe}_{embedding_safe}"
        return vision_safe
    
    def get_db_path(self, vision_model: str, embedding_model: str = None, db_type: str = "descriptions") -> str:
        """Get database path for a specific model pair and type"""
        db_name = self.get_db_name(vision_model, embedding_model)
        return str(self.data_dir / f"{db_name}_{db_type}.db")
    
    def init_descriptions_db(self, vision_model: str, embedding_model: str = None):
        """Initialize descriptions database"""
        db_path = self.get_db_path(vision_model, embedding_model, "descriptions")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                image_name TEXT,
                image_hash TEXT,
                image_data BLOB,
                description TEXT,
                prompt TEXT,
                context_size INTEGER,
                vision_model TEXT,
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    
    def get_embedding_model_from_db(self, db_path: Path) -> Optional[str]:
        """Extract embedding model from the DB metadata"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding_model FROM descriptions LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0]
        except Exception as e:
            print(f"[ERROR] Failed to extract embedding model: {e}")
        return None
    
    def save_description(self, vision_model: str, embedding_model: str, image_path: str, 
                        description: str, prompt: str, context_size: int):
        """Save image description"""
        self.init_descriptions_db(vision_model, embedding_model)
        db_path = self.get_db_path(vision_model, embedding_model, "descriptions")
        
        # Calculate image hash
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_hash = hashlib.md5(image_data).hexdigest()
        
        image_name = Path(image_path).name
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO descriptions 
            (image_path, image_name, image_hash, image_data, description, prompt, context_size, vision_model, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (image_path, image_name, image_hash, image_data, description, prompt, context_size, vision_model, embedding_model))
        conn.commit()
        conn.close()
    
    def save_embeddings(self, vision_model: str, embedding_model: str, embeddings_data: Dict):
        """Merge and save embeddings to pickle file (preserve old data)"""
        db_name = self.get_db_name(vision_model, embedding_model)
        embeddings_path = self.data_dir / f"{db_name}_embeddings.pkl"

        # Load existing embeddings
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                existing = pickle.load(f)
        else:
            existing = {}

        # Merge: overwrite only the keys we just processed
        existing.update(embeddings_data)

        # Save merged result
        with open(embeddings_path, "wb") as f:
            pickle.dump(existing, f)
    
    def load_embeddings(self, vision_model: str, embedding_model: str) -> Dict:
        """Load embeddings from pickle file"""
        db_name = self.get_db_name(vision_model, embedding_model)
        embeddings_path = self.data_dir / f"{db_name}_embeddings.pkl"
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                return pickle.load(f)
        return {}
        
    def keyword_search(self, db_path: Path, query: str) -> List[Tuple]:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_path, image_name, description, created_at
            FROM descriptions
            WHERE description LIKE ? OR image_name LIKE ?
            ORDER BY created_at DESC
        """, (f"%{query}%", f"%{query}%"))
        results = cursor.fetchall()
        conn.close()
        return results
    
    def search_descriptions(self, db_name: str, query: str, threshold: float) -> List[Tuple]:
        """Search descriptions using semantic similarity"""
        db_path = self.data_dir / f"{db_name}_descriptions.db"
        embeddings_path = self.data_dir / f"{db_name}_embeddings.pkl"

        if not db_path.exists():
            print(f"[WARN] DB file not found: {db_path}")
            return []

        # Load saved embeddings
        if not embeddings_path.exists():
            print(f"[WARN] Embeddings file not found: {embeddings_path}")
            return self.keyword_search(db_path, query)

        try:
            with open(embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load embeddings: {e}")
            return self.keyword_search(db_path, query)

        if not embeddings:
            print("[INFO] No embeddings found, falling back to keyword search.")
            return self.keyword_search(db_path, query)

        embedding_model = self.get_embedding_model_from_db(db_path)
        if not embedding_model:
            print("[ERROR] Could not determine embedding model from DB.")
            return self.keyword_search(db_path, query)

        print(f"[DEBUG] Using embedding model from DB: {embedding_model}")
        # Embed the query
        try:
            query_embedding = self.ollama_client.generate_embedding(embedding_model, query)
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            return self.keyword_search(db_path, query)

        if not query_embedding:
            print(f"[DEBUG] Query: '{query}' (len={len(query)})")
            print(f"[DEBUG] Calling generate_embedding with model='{embedding_model}' and query='{query}'")
            print("[ERROR] Query embedding is None.")
            return self.keyword_search(db_path, query)

        # Compute cosine similarity
        image_paths = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[path] for path in image_paths])
        query_vector = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity_numpy(query_vector, embedding_matrix)

        # Rank and filter (semantic order)
        ranked = sorted(zip(image_paths, similarities), key=lambda x: x[1], reverse=True)
        top_matches = [(path, score) for path, score in ranked if score > threshold]
        print(f"[DEBUG] Found {len(top_matches)} matches above threshold.")

        if not top_matches:
            best_path, best_score = ranked[0]
            print(f"[INFO] No semantic matches found. Best score was {best_score:.3f}, falling back to keyword search.")
            return self.keyword_search(db_path, query)

        # Query database for those image paths
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in top_matches)
        cursor.execute(f"""
            SELECT image_path, image_name, description, created_at
            FROM descriptions
            WHERE image_path IN ({placeholders})
        """, [p for p, _ in top_matches])
        results = cursor.fetchall()
        conn.close()

        # Reorder results by similarity score (not DB order)
        results_dict = {row[0]: row for row in results}
        ordered_results = [results_dict[p] for p, _ in top_matches if p in results_dict]

        return ordered_results

    
    
    def get_all_descriptions(self, db_name: str) -> List[Tuple]:
        """Get all descriptions from database"""
        db_path = self.data_dir / f"{db_name}_descriptions.db"
        if not db_path.exists():
            return []
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_path, image_name, description, vision_model, embedding_model, created_at
            FROM descriptions
            ORDER BY created_at DESC
        """)
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_available_databases(self) -> List[str]:
        """Get list of available model databases"""
        databases = []
        for file in self.data_dir.glob("*_descriptions.db"):
            db_name = file.stem.replace("_descriptions", "")
            databases.append(db_name)
        return databases
    def get_existing_paths(self, vision_model: str, embedding_model: str = None) -> set:
        """Return a set of all image paths already stored in the DB."""
        self.init_descriptions_db(vision_model, embedding_model)
        db_path = self.get_db_path(vision_model, embedding_model, "descriptions")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM descriptions")
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        return existing
    
    def get_image_data_from_db(self, db_name: str, image_path: str) -> Optional[bytes]:
        """Get image data from specific database"""
        db_path = self.data_dir / f"{db_name}_descriptions.db"
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT image_data FROM descriptions WHERE image_path = ?", (image_path,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None



def cosine_similarity_numpy(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Normalize query and matrix
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # Compute dot products
    return np.dot(query_norm, matrix_norm.T)[0]


class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    
    def __init__(self, ollama_client, db_manager, image_paths, vision_model, 
                 embedding_model, prompt, context_size):
        super().__init__()
        self.ollama_client = ollama_client
        self.db_manager = db_manager
        self.image_paths = image_paths
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.prompt = prompt
        self.context_size = context_size
        self.should_stop = False
    
    def stop(self):
        self.should_stop = True
        
        
    def shorten_filename(filename: str, max_length: int = 40) -> str:
        name = Path(filename).name
        if len(name) <= max_length:
            return name

        stem = Path(name).stem
        suffix = Path(name).suffix

        # Reserve space for suffix and ellipsis
        max_stem_len = max_length - len(suffix) - 3
        if max_stem_len <= 0:
            return "..." + suffix  # fallback if suffix is too long

        return stem[:max_stem_len] + "..." + suffix
    
    def run(self):
        try:
            total_images = len(self.image_paths)
            descriptions = {}
            embeddings = {}
            batch_size = 50
            temp_batch = {}
            MAX_RETRIES = 3
            failed_descriptions = []

            processing_start_time = time.time()
            
            
            existing_paths = self.db_manager.get_existing_paths(self.vision_model, self.embedding_model)

            # Filter out images that are already processed
            unprocessed_paths = [p for p in self.image_paths if p not in existing_paths]

            if not unprocessed_paths:
                print("[INFO] All images already have descriptions in DB — skipping processing.")
                self.progress.emit(100)
                self.finished.emit({"descriptions": {}, "embeddings": {}})
                return

            self.image_paths = unprocessed_paths
            total_images = len(self.image_paths)

            print(f"[INFO] {len(existing_paths)} images already processed — skipping them.")
            print(f"[INFO] {total_images} images left to process.")
            

            def describe_task(image_path):
                description = self.ollama_client.generate_description(
                    self.vision_model, image_path, self.prompt, self.context_size
                )
                return (image_path, description)

            # Phase 1: Generate descriptions in parallel
            with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
                for i, result in enumerate(executor.map(describe_task, self.image_paths)):
                    if self.should_stop:
                        self.stopped.emit()
                        return

                    image_path, description = result
                    name = Path(image_path).name
                    if len(name) > 40:
                        name = name[:29] + "..." + name[-8:]

                    self.status.emit(f"Processing image {i+1}/{total_images}: {name}")

                    if description and not description.startswith("Error"):
                        descriptions[image_path] = description
                        temp_batch[image_path] = description
                    else:
                        print(f"[ERROR] Processing description failed for {name}: {description}")
                        failed_descriptions.append(image_path)


                    # Sync every batch_size
                    if len(temp_batch) >= batch_size or i == total_images - 1:
                        for path, desc in temp_batch.items():
                            self.db_manager.save_description(
                                self.vision_model, self.embedding_model, path,
                                desc, self.prompt, self.context_size
                            )
                        temp_batch.clear()

                    progress = int((i + 1) / total_images * 50)
                    self.progress.emit(progress)

            print(f"[INFO] Total images: {total_images}")
            print(f"[INFO] Successful descriptions: {len(descriptions)}")
            print(f"[INFO] Failed descriptions: {len(failed_descriptions)}")
            
            #Retry
            retry_count = 0

            while failed_descriptions and retry_count < MAX_RETRIES:
                retry_count += 1
                print(f"[INFO] Retry pass {retry_count} for {len(failed_descriptions)} failed images")

                retry_list = failed_descriptions.copy()
                failed_descriptions.clear()

                for image_path in retry_list:
                    if self.should_stop:
                        self.stopped.emit()
                        return
                    description = self.ollama_client.generate_description(
                        self.vision_model, image_path, self.prompt, self.context_size
                    )

                    name = Path(image_path).name
                    if len(name) > 40:
                        name = name[:29] + "..." + name[-8:]

                    if description and not description.startswith("Error"):
                        descriptions[image_path] = description
                        temp_batch[image_path] = description
                        print(f"[RETRY SUCCESS] {name}")
                    else:
                        print(f"[RETRY FAILED] {name}: {description or 'No response'}")
                        failed_descriptions.append(image_path)



            # Phase 2: Generate embeddings in parallel
            if self.embedding_model:
                def embed_task(path, desc):
                    emb = self.ollama_client.generate_embedding(self.embedding_model, desc)
                    return (path, emb)

                futures = []
                paths = list(descriptions.keys())
                with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
                    for path, desc in descriptions.items():
                        futures.append(executor.submit(embed_task, path, desc))

                    for i, future in enumerate(as_completed(futures)):
                        if self.should_stop:
                            self.stopped.emit()
                            return

                        try:
                            path, emb = future.result()
                            if emb:
                                embeddings[path] = emb
                        except Exception as e:
                            print(f"[ERROR] Embedding failed for {path}: {e}")

                        # Emit progress (second half: 50–100%)
                        progress = 50 + int((i + 1) / len(futures) * 50)
                        self.progress.emit(progress)

                        # Emit status text
                        name = os.path.basename(paths[i])
                        if len(name) > 40:
                            name = name[:29] + "..." + name[-8:]
                        self.status.emit(f"Embedding {i+1}/{len(futures)}: {name}")



                self.db_manager.save_embeddings(self.vision_model, self.embedding_model, embeddings)

            self.progress.emit(100)
            self.status.emit(f"Processing completed in {time.time() - processing_start_time:.2f} seconds!")
            print(f"[Startup] Processing took {time.time() - processing_start_time:.2f} seconds")
            self.finished.emit({"descriptions": descriptions, "embeddings": embeddings})

        except Exception as e:
            self.error.emit(str(e))


class ImageWidget(QLabel):
    def __init__(self, image_path: str, logger=None):
        super().__init__()
        self.image_path = image_path
        self.logger = logger  # Reference to MainWindow or logging method
        self.setFixedSize(200, 200)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel:hover {
                border-color: #0078d4;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.load_image()
    
    def load_image(self):
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    190, 190, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
            else:
                self.setText("Invalid\nImage")
        except:
            self.setText("Error\nLoading")
    
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        pixmap = QPixmap(self.image_path)
        if pixmap.isNull():
            self.setToolTip("Failed to copy image")
            if self.logger:
                self.logger(f"Failed to copy image: {Path(self.image_path).name}")
            return

        # Wrap QByteArray in QBuffer
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()

        mime_data = QMimeData()
        mime_data.setData("image/png", byte_array)

        mime_data.setUrls([QUrl.fromLocalFile(self.image_path)])

        QApplication.clipboard().setMimeData(mime_data)

        self.setToolTip(f"Copied image to clipboard: {Path(self.image_path).name}")
        if self.logger:
            self.logger(f"Copied image to clipboard: {Path(self.image_path).name}")


class SearchThread(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, db_manager, db_name, query, threshold):
        super().__init__()
        self.db_manager = db_manager
        self.db_name = db_name
        self.query = query
        self.threshold = threshold

    def run(self):
        results = self.db_manager.search_descriptions(self.db_name, self.query, self.threshold)
        self.result_ready.emit(results)


class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)  # Emit image_data (bytes)

    def __init__(self, image_data: bytes, parent=None):
        super().__init__(parent)
        self.image_data = image_data

    def mousePressEvent(self, event):
        self.clicked.emit(self.image_data)


class PostInitWorker(QObject):
    finished = pyqtSignal()
    models_ready = pyqtSignal(list, list)
    databases_ready = pyqtSignal(list)

    def __init__(self, ollama_client, db_manager, split_models_fn):
        super().__init__()
        self.ollama_client = ollama_client
        self.db_manager = db_manager
        self.split_models_by_families = split_models_fn

    def run(self):
        postinit_start_time = time.time()
        models = self.ollama_client.get_models()
        if models:
            vision, embedding = self.split_models_by_families(models)
            self.models_ready.emit(vision, embedding)

        databases = self.db_manager.get_available_databases()
        print(f"[Startup] Post Init took {time.time() - postinit_start_time:.2f} seconds")
        self.databases_ready.emit(databases)
        self.finished.emit()


class MainWindow(QMainWindow):
    
    models_loaded_signal = pyqtSignal(list, list)
    databases_loaded_signal = pyqtSignal(list)

    
    def __init__(self):
        super().__init__()
        
        self.processing_worker = None
        self.models_loaded = False
        self.ollama_client = OllamaClient()
        self.db_manager = DatabaseManager(ollama_client=self.ollama_client)
        self.init_ui()
        self.apply_dark_theme()
        
        self.models_loaded_signal.connect(self.update_model_combos)
        self.databases_loaded_signal.connect(self.update_database_combos)

        
        # Defer heavy setup
        QTimer.singleShot(50, self.start_post_init_tasks)


    def start_post_init_tasks(self):
        
        self.thread = QThread()
        self.worker = PostInitWorker(
            ollama_client=self.ollama_client,
            db_manager=self.db_manager,
            split_models_fn=self.split_models_by_families
        )
        self.worker.moveToThread(self.thread)

        # Connect worker signals to MainWindow slots
        self.worker.models_ready.connect(self.update_model_combos)
        self.worker.databases_ready.connect(self.update_database_combos)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        print("⏳ Loading models and databases...")
      
    
    def init_ui(self):
        self.setWindowTitle("Image Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        main_layout.addWidget(right_panel, 2)
        
    def copy_image_to_clipboard(self, image_data):
        if not image_data:
            self.log_verbose("No image data to copy")
            return

        byte_array = QByteArray(image_data)
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        pixmap.save(buffer, "PNG")
        buffer.close()

        mime_data = QMimeData()
        mime_data.setData("image/png", byte_array)

        QApplication.clipboard().setMimeData(mime_data)
        self.log_verbose("Copied image to clipboard from database")


    def split_models_by_families(self, models: List[dict]) -> Tuple[List[str], List[str]]:
        vision_keywords = {"clip", "llava", "blip", "vision", "mllama", "qwen25vl"}
        embedding_keywords = {"embed", "nomic", "e5", "bge", "text-embedding", "bert"}

        vision: List[str] = []
        embedding: List[str] = []

        for model in models:
            # Model name (safe)
            name = (model.get("name") or "").lower()

            # Details may be None
            details = model.get("details") or {}

            # Families may be None or non-list
            families = details.get("families")
            if not isinstance(families, list):
                families = []

            families = [f.lower() for f in families if isinstance(f, str)]

            # Vision model if any family matches
            if any(f in vision_keywords for f in families):
                vision.append(model.get("name", ""))

            # Embedding model if any family matches or name contains 'embed'
            elif any(f in embedding_keywords for f in families) or "embed" in name:
                embedding.append(model.get("name", ""))

            # Optional debug
            # else:
            #     print(f"[Unclassified] {model.get('name')} → families: {families}")

        return vision, embedding

    def update_model_combos(self, vision_models: List[str], embedding_models: List[str]):
        current_vision = self.vision_model_combo.currentText()
        self.vision_model_combo.clear()
        self.vision_model_combo.addItems(vision_models)
        if current_vision in vision_models:
            self.vision_model_combo.setCurrentText(current_vision)

        current_embedding = self.embedding_model_combo.currentText()
        self.embedding_model_combo.clear()
        self.embedding_model_combo.addItems(embedding_models)
        if current_embedding in embedding_models:
            self.embedding_model_combo.setCurrentText(current_embedding)

        self.log_verbose(f"Loaded {len(vision_models)} vision models from Ollama")
        print(f"Loaded {len(vision_models)} vision models from Ollama")
        self.log_verbose(f"Loaded {len(embedding_models)} embedding models from Ollama")
        print(f"Loaded {len(embedding_models)} embedding models from Ollama")

    def update_database_combos(self, databases):
        self.search_db_combo.clear()
        self.search_db_combo.addItems(databases)

        self.desc_db_combo.clear()
        self.desc_db_combo.addItems(databases)

        self.log_verbose(f"Found {len(databases)} databases")
        
    def update_threshold_label(self, value):
        threshold = value / 100.0
        self.threshold_label.setText(f"Similarity Threshold: {threshold:.2f}")
    
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Directory selection
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout(dir_group)
        
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setPlaceholderText("Select directory containing images...")
        dir_layout.addWidget(self.dir_path_edit)
        
        dir_btn = QPushButton("Browse Directory")
        dir_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(dir_btn)
        
        layout.addWidget(dir_group)
        
        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Vision Model:"))
        self.vision_model_combo = QComboBox()
        self.vision_model_combo.addItems(["Loading..."])
        model_layout.addWidget(self.vision_model_combo)
        
        model_layout.addWidget(QLabel("Embedding Model:"))
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.addItems(["Loading..."])
        model_layout.addWidget(self.embedding_model_combo)
        
        layout.addWidget(model_group)
        
        # Processing parameters
        params_group = QGroupBox("Processing Parameters")
        params_layout = QVBoxLayout(params_group)
        
        params_layout.addWidget(QLabel("Context Size:"))
        self.context_size_combo = QComboBox()
        self.context_size_combo.addItems(["256", "512", "1024", "2048", "4096", "8192"])
        self.context_size_combo.setCurrentText("2048")
        params_layout.addWidget(self.context_size_combo)
        
        params_layout.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText("Describe this image in detail, including objects, colors, composition, and any text visible.")
        self.prompt_edit.setMaximumHeight(100)
        params_layout.addWidget(self.prompt_edit)
        
        layout.addWidget(params_group)
        
        # Processing controls
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)
        
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        process_layout.addWidget(self.process_btn)
        
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)

        self.stop_btn.setEnabled(False)
        process_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        process_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        process_layout.addWidget(self.status_label)
        
        layout.addWidget(process_group)
        
        # Search section
        search_group = QGroupBox("Search & Database")
        search_layout = QVBoxLayout(search_group)
        
        search_layout.addWidget(QLabel("Search Database:"))
        self.search_db_combo = QComboBox()
        search_layout.addWidget(self.search_db_combo)
        
        search_layout.addWidget(QLabel("Search Query:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Enter search terms...")
        self.search_edit.returnPressed.connect(self.search_descriptions)
        search_layout.addWidget(self.search_edit)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(40)  # Default to 0.4
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)

        self.threshold_label = QLabel("Similarity Threshold: 0.40")

        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        search_layout.addWidget(self.threshold_label)
        search_layout.addWidget(self.threshold_slider)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_descriptions)
        search_layout.addWidget(self.search_btn)
        
        refresh_db_btn = QPushButton("Refresh Databases")
        refresh_db_btn.clicked.connect(self.refresh_databases)
        search_layout.addWidget(refresh_db_btn)
        
        layout.addWidget(search_group)
        
        # Verbose output
        verbose_group = QGroupBox("Verbose Output")
        verbose_layout = QVBoxLayout(verbose_group)
        
        self.verbose_text = QTextEdit()
        self.verbose_text.setMaximumHeight(150)
        self.verbose_text.setReadOnly(True)
        verbose_layout.addWidget(self.verbose_text)
        
        layout.addWidget(verbose_group)
        
        layout.addStretch()
        return panel
    
    def create_results_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Image explorer tab
        self.image_explorer = self.create_image_explorer()
        self.tab_widget.addTab(self.image_explorer, "Image Explorer")
        
        # Search results tab
        self.search_results = self.create_search_results()
        self.tab_widget.addTab(self.search_results, "Search Results")
        
        # Descriptions explorer tab
        self.descriptions_explorer = self.create_descriptions_explorer()
        self.tab_widget.addTab(self.descriptions_explorer, "Descriptions")
        
        return panel
    
    def create_image_explorer(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scroll area for images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.image_container = QWidget()
        self.image_grid_layout = QGridLayout(self.image_container)
        self.image_grid_layout.setSpacing(10)
        
        scroll_area.setWidget(self.image_container)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_search_results(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.search_table = QTableWidget()
        self.search_table.setColumnCount(4)
        self.search_table.setHorizontalHeaderLabels(["Image", "Image Path", "Description", "Date"])
        self.search_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.search_table.setAlternatingRowColors(True)
        self.search_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.search_table)
        return widget
    
    def create_descriptions_explorer(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        
        db_select_layout = QHBoxLayout()
        db_select_layout.addWidget(QLabel("Database:"))

        self.desc_db_combo = QComboBox()
        db_select_layout.addWidget(self.desc_db_combo)
        self.desc_db_combo.setFixedWidth(300) 
        self.load_desc_button = QPushButton("Load Descriptions")
        self.load_desc_button.setFixedWidth(150) 
        self.load_desc_button.clicked.connect(self.load_descriptions)
        db_select_layout.addStretch()
        db_select_layout.addWidget(self.load_desc_button)

        
        layout.addLayout(db_select_layout)


        # Descriptions table
        self.descriptions_table = QTableWidget()
        self.descriptions_table.setColumnCount(6)
        self.descriptions_table.setHorizontalHeaderLabels([
            "Image", "Description", "Vision Model", "Embedding Model", "Date", "Path"
        ])
        self.descriptions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.descriptions_table.setAlternatingRowColors(True)
        self.descriptions_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.descriptions_table)
        return widget
    
    def apply_dark_theme(self):
        dark_palette = QPalette()
        
        # Window colors
        dark_palette.setColor(QPalette.Window, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        
        # Base colors
        dark_palette.setColor(QPalette.Base, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
        
        # Text colors
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
        
        # Button colors
        dark_palette.setColor(QPalette.Button, QColor(55, 55, 55))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        
        # Highlight colors
        dark_palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Disabled colors
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        
        self.setPalette(dark_palette)
        
        # Additional styling
        self.setStyleSheet("""
            QWidget {
                background-color: #232323;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QComboBox, QLineEdit, QSpinBox {
                padding: 5px;
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #0078d4;
                width: 20px;
            }
            QComboBox::down-arrow {
                border: none;
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #ffffff;
                selection-background-color: #0078d4;
                border: 1px solid #555;
            }
            QTextEdit {
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #2b2b2b;
                padding: 5px;
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid #555;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            QTabWidget::pane {
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #555;
                selection-background-color: #0078d4;
            }
            QTableWidget::item {
                border: none;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #555;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #777;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 15px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #0078d4;
            }
            QScrollArea {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def load_models_once(self) -> Optional[Tuple[List[str], List[str]]]:
        if self.models_loaded:
            return None

        models = self.ollama_client.get_models()
        if models:
            vision_models, embedding_models = self.split_models_by_families(models)
            self.models_loaded = True
            self.log_verbose(f"Loaded {len(models)} models from Ollama")
            self.log_verbose(f"Vision models: {len(vision_models)}, Embedding models: {len(embedding_models)}")
            return vision_models, embedding_models
        else:
            self.log_verbose("Could not connect to Ollama or no models available")
            return None
    
    def refresh_databases(self) -> List[str]:
        self.start_post_init_tasks()

    
    def browse_directory(self):
        """Browse for image directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.dir_path_edit.setText(directory)
            self.load_images_preview(directory)
    
    def get_image_files(self, directory: str) -> List[Path]:
        """Get unique image files from directory"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        image_files = []
        seen_files = set()
        
        directory_path = Path(directory)
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Use absolute path to avoid duplicates
                abs_path = file_path.resolve()
                if abs_path not in seen_files:
                    seen_files.add(abs_path)
                    image_files.append(abs_path)
        
        return image_files
    
    def load_images_preview(self, directory):
        """Load preview of images in directory"""
        # Clear existing images
        for i in reversed(range(self.image_grid_layout.count())):
            child = self.image_grid_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Get unique image files
        image_paths = self.get_image_files(directory)
        
        # Display first 20 images as preview
        cols = 4
        for i, image_path in enumerate(image_paths[:2000]):
            row = i // cols
            col = i % cols
            
            image_widget = ImageWidget(str(image_path), logger=self.log_verbose)
            image_widget.setToolTip(f"Click to copy path: {image_path.name}")
            self.image_grid_layout.addWidget(image_widget, row, col)
        
        self.log_verbose(f"Found {len(image_paths)} unique images in directory")
    
    def start_processing(self):
        """Start processing images"""
        directory = self.dir_path_edit.text()
        if not directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first")
            return
        
        vision_model = self.vision_model_combo.currentText()
        embedding_model = self.embedding_model_combo.currentText()
        
        if not vision_model:
            QMessageBox.warning(self, "Warning", "Please select a vision model")
            return
        
        # Get unique image files
        image_paths = self.get_image_files(directory)
        
        if not image_paths:
            QMessageBox.warning(self, "Warning", "No images found in directory")
            return
        
        self.status_label.setText("Processing starting...")
        print("Processing starting...")
        # Start processing
        prompt = self.prompt_edit.toPlainText()
        context_size = int(self.context_size_combo.currentText())
        
        self.processing_worker = ProcessingWorker(
            self.ollama_client, self.db_manager, [str(p) for p in image_paths],
            vision_model, embedding_model, prompt, context_size
        )
        
        self.processing_worker.progress.connect(self.progress_bar.setValue)
        self.processing_worker.status.connect(self.status_label.setText)
        self.processing_worker.status.connect(self.log_verbose)
        self.processing_worker.finished.connect(self.processing_finished)
        self.processing_worker.error.connect(self.processing_error)
        self.processing_worker.stopped.connect(self.on_processing_stopped)
        self.processing_worker.start()
        
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.log_verbose(f"Started processing {len(image_paths)} images with {vision_model}")
    
    def stop_processing(self):
        if self.processing_worker:
            self.processing_worker.stop()  # Just request stop — don't wait

        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping...")
        self.log_verbose("Stop requested")

    def on_processing_stopped(self):
        self.process_btn.setEnabled(True)
        self.status_label.setText("Processing stopped")
        self.log_verbose("Processing fully stopped")

    
    def processing_finished(self, results):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        descriptions = results.get("descriptions", {})
        embeddings = results.get("embeddings", {})
        
        self.log_verbose(f"Processing completed: {len(descriptions)} descriptions, {len(embeddings)} embeddings")
        self.refresh_databases()
        
    
    def processing_error(self, error_msg):
        """Handle processing error"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_verbose(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Processing failed: {error_msg}")
    
    def display_search_results(self, results, query):
        db_name = self.search_db_combo.currentText()
        self.search_table.setRowCount(len(results))
        self.search_table.setAlternatingRowColors(False)

        v_header = self.search_table.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.Fixed)

        header = self.search_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.search_table.setColumnWidth(0, 150)

        for row, (image_path, image_name, description, date) in enumerate(results):
            self.search_table.setRowHeight(row, 150)

            # Image preview
            image_data = self.db_manager.get_image_data_from_db(db_name, image_path)
            image_label = ClickableLabel(image_data)
            if image_data:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("No Image")
                image_label.setAlignment(Qt.AlignCenter)

            image_label.setAlignment(Qt.AlignCenter)
            image_label.clicked.connect(self.copy_image_to_clipboard)
            self.search_table.setCellWidget(row, 0, image_label)

            # Path
            self.search_table.setItem(row, 1, QTableWidgetItem(image_path))

            # Description
            desc_preview = description[:250] + "..." if len(description) > 250 else description
            self.search_table.setItem(row, 2, QTableWidgetItem(desc_preview))

            # Date
            self.search_table.setItem(row, 3, QTableWidgetItem(date))

        self.tab_widget.setCurrentIndex(1)
        self.log_verbose(f"Found {len(results)} results for query: '{query}'")
    
    def search_descriptions(self):
        query = self.search_edit.text()
        db_name = self.search_db_combo.currentText()
        threshold = self.threshold_slider.value() / 100.0

        if not query or not db_name:
            QMessageBox.information(self, "Info", "Please enter a search query and select a database")
            return

        self.search_btn.setEnabled(False)
        self.log_verbose(f"Searching: '{query}'")

        self.search_thread = SearchThread(self.db_manager, db_name, query, threshold)
        self.search_thread.result_ready.connect(lambda results: self.display_search_results(results, query))
        self.search_thread.finished.connect(lambda: self.search_btn.setEnabled(True))
        self.search_thread.start()


    
    def load_descriptions(self):
        descload_start_time = time.time()

        """Load all descriptions from selected database"""
        db_name = self.desc_db_combo.currentText()
        if not db_name:
            return
        
        results = self.db_manager.get_all_descriptions(db_name)
        
        self.descriptions_table.setRowCount(len(results))
        self.descriptions_table.setAlternatingRowColors(False)
        
        v_header = self.descriptions_table.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.Fixed)

        
        header = self.descriptions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Lock column 0
        self.descriptions_table.setColumnWidth(0, 100)      # Force width

        for row, (image_path, image_name, description, vision_model, embedding_model, date) in enumerate(results):
            self.descriptions_table.setRowHeight(row, 100)

            # Image preview in column
            image_data = self.db_manager.get_image_data_from_db(db_name, image_path)
            image_label = ClickableLabel(image_data)
            if image_data:
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                scaled_pixmap = pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("No Image")
                image_label.setAlignment(Qt.AlignCenter)

            image_label.setAlignment(Qt.AlignCenter)
            image_label.clicked.connect(self.copy_image_to_clipboard)
            self.descriptions_table.setCellWidget(row, 0, image_label)

            # Truncated description in column
            desc_preview = description[:250] + "..." if len(description) > 250 else description
            self.descriptions_table.setItem(row, 1, QTableWidgetItem(desc_preview))

            # Vision model in column
            self.descriptions_table.setItem(row, 2, QTableWidgetItem(vision_model or "N/A"))

            # Embedding model in column
            self.descriptions_table.setItem(row, 3, QTableWidgetItem(embedding_model or "N/A"))

            # Date in column
            self.descriptions_table.setItem(row, 4, QTableWidgetItem(date))

            # Image path in column
            self.descriptions_table.setItem(row, 5, QTableWidgetItem(image_path))
            
        self.log_verbose(f"Loaded {len(results)} descriptions from database: {db_name}")
        print(f"[Startup] Desc Load took {time.time() - descload_start_time:.2f} seconds")
        self.log_verbose(f"[Startup] Desc Load took {time.time() - descload_start_time:.2f} seconds")

    
    def log_verbose(self, message):
        """Add message to verbose output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.verbose_text.append(f"[{timestamp}] {message}")
        
        # Keep only last 100 lines
        document = self.verbose_text.document()
        if document.blockCount() > 100:
            cursor = self.verbose_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 
                              document.blockCount() - 100)
            cursor.removeSelectedText()
        
        # Auto-scroll to bottom
        cursor = self.verbose_text.textCursor()
        cursor.movePosition(cursor.End)
        self.verbose_text.setTextCursor(cursor)


def resource_path(relative_path):
  if hasattr(sys, '_MEIPASS'):
      return os.path.join(sys._MEIPASS, relative_path)
  return os.path.join(os.path.abspath('.'), relative_path)

def main():
    startup_start_time = time.time()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  #Dark theme support
    app.setWindowIcon(QIcon(resource_path('./icon.ico')))

    window = MainWindow()

    window.show()
    
    print(f"[Startup] Init took {time.time() - startup_start_time:.2f} seconds")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()