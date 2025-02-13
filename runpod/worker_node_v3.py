#!/usr/bin/python3 worker_node_v3.py
#audio-workernode/runpod/worker_node_v3.py

import os
import json
import time
import requests
import logging
import traceback
from typing import Dict, Any, Optional
from configuration_manager import ConfigurationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("WorkerNode")

class AudioTranscriptionWorker:
    def __init__(self):
        """Initialize API-based Transcription Worker"""
        try:
            self.config = ConfigurationManager("worker_api.config.yaml").get_config()
            self.worker_id = self.config["worker"]["name"] + self._get_unique_id()
            self.orchestrator_url = self.config["orchestrator"]["url"]
            self.api_token = self.config["secrets"]["api_token"]
            self.transcription_api_url = self.config["transcription"]["api_url"]
            self.keep_running = True
        except Exception as e:
            logger.critical(f"Failed to initialize worker: {e}")
            raise SystemExit("Cannot start worker without configuration")

    def _get_unique_id(self) -> str:
        """Generate unique identifier for worker."""
        return f"-pod-{os.environ.get('RUNPOD_POD_ID', 'unknown')}"

    def get_task(self) -> Optional[Dict[str, Any]]:
        """Fetch a new task from the orchestrator."""
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            response = requests.get(f"{self.orchestrator_url}/get-task", headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 204:
                return None
            else:
                logger.error(f"Failed to get task: {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error getting task: {e}")
            return None

    def transcribe_audio_via_api(self, audio_path: str) -> Optional[str]:
        """Send audio to the API-based transcription service."""
        try:
            with open(audio_path, "rb") as f:
                files = {"file": f}
                data = {"language": "en"}
                headers = {"Authorization": f"Bearer {self.api_token}"}
                response = requests.post(
                    self.transcription_api_url, files=files, data=data, headers=headers, timeout=30
                )

            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                logger.error(f"API transcription failed: {response.status_code} {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception during API transcription: {e}")
            traceback.print_exc()
            return None

    def process_task(self, task: Dict[str, Any]) -> bool:
        """Process a single transcription task."""
        task_id = task["task_id"]
        presigned_get_url = task["presigned_get_url"]
        presigned_put_url = task["presigned_put_url"]
        filename = os.path.basename(task["object_key"])
        local_audio_path = os.path.join(self.config["storage"]["download_folder"], filename)
        local_transcript_path = f"{local_audio_path}.txt"

        try:
            if not self._download_file(presigned_get_url, local_audio_path):
                return self._fail_task(task_id, "Download failed")

            transcription = self.transcribe_audio_via_api(local_audio_path)
            if not transcription:
                return self._fail_task(task_id, "Transcription failed")

            if not self._upload_transcription(transcription, local_transcript_path, presigned_put_url):
                return self._fail_task(task_id, "Upload failed")

            self._update_task_status(task_id, "Completed")
            return True
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            traceback.print_exc()
            return self._fail_task(task_id, "Unexpected error")

    def _download_file(self, presigned_url: str, local_path: str) -> bool:
        """Download an audio file from S3."""
        try:
            response = requests.get(presigned_url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)
                return True
            return False
        except requests.RequestException as e:
            logger.error(f"Download error: {e}")
            return False

    def _upload_transcription(self, transcription: str, local_path: str, presigned_url: str) -> bool:
        """Save and upload transcription."""
        try:
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            with open(local_path, "rb") as f:
                response = requests.put(presigned_url, data=f, headers={"Content-Type": "text/plain"}, timeout=30)
            return response.status_code in [200, 204]
        except requests.RequestException as e:
            logger.error(f"Upload error: {e}")
            return False

    def _update_task_status(self, task_id: str, status: str):
        """Notify orchestrator of task status."""
        requests.post(f"{self.orchestrator_url}/update-task-status", json={"task_id": task_id, "status": status})

    def _fail_task(self, task_id: str, reason: str) -> bool:
        """Mark task as failed and return False."""
        self._update_task_status(task_id, "Failed")
        return False

    def run(self):
        """Main loop to process transcription tasks."""
        while self.keep_running:
            task = self.get_task()
            if task:
                self.process_task(task)
            else:
                time.sleep(self.config["performance"]["poll_interval"])

if __name__ == "__main__":
    worker = AudioTranscriptionWorker()
    worker.run()
