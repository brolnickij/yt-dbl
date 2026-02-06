"""Step 1: Download video and audio from YouTube via yt-dlp."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

from yt_dbl.pipeline.base import DownloadError, PipelineStep, StepValidationError
from yt_dbl.schemas import PipelineState, StepName, VideoMeta
from yt_dbl.utils.audio import extract_audio
from yt_dbl.utils.logging import create_progress, log_info

if TYPE_CHECKING:
    from pathlib import Path

VIDEO_FILENAME = "video.mp4"
AUDIO_FILENAME = "audio.wav"


class DownloadStep(PipelineStep):
    name = StepName.DOWNLOAD
    description = "Download video + extract audio"

    def validate_inputs(self, state: PipelineState) -> None:
        if not state.url:
            raise StepValidationError("No URL provided")

    def run(self, state: PipelineState) -> PipelineState:
        video_path = self.step_dir / VIDEO_FILENAME
        audio_path = self.step_dir / AUDIO_FILENAME

        # 1. Fetch metadata
        log_info("Fetching video metadata...")
        meta = self._fetch_metadata(state.url)
        state.meta = VideoMeta(
            video_id=state.video_id,
            title=meta.get("title", ""),
            channel=meta.get("channel", meta.get("uploader", "")),
            duration=float(meta.get("duration", 0)),
            url=state.url,
        )
        log_info(f"Title: {state.meta.title}")
        log_info(f"Channel: {state.meta.channel}")
        log_info(f"Duration: {state.meta.duration:.0f}s")

        # 2. Download video
        if not video_path.exists():
            self._download_video(state.url, video_path)
        else:
            log_info("Video file already exists, skipping download")

        # 3. Extract audio as WAV (48kHz mono)
        if not audio_path.exists():
            log_info("Extracting audio (48kHz mono WAV)...")
            extract_audio(
                video_path,
                audio_path,
                sample_rate=self.settings.sample_rate,
                mono=True,
            )
        else:
            log_info("Audio file already exists, skipping extraction")

        # Verify outputs exist
        if not video_path.exists():
            raise DownloadError("Video file was not created")
        if not audio_path.exists():
            raise DownloadError("Audio file was not created")

        video_size_mb = video_path.stat().st_size / (1024 * 1024)
        audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
        log_info(f"Video: {video_size_mb:.1f} MB, Audio: {audio_size_mb:.1f} MB")

        result = state.get_step(self.name)
        result.outputs = {
            "video": VIDEO_FILENAME,
            "audio": AUDIO_FILENAME,
        }

        return state

    def _fetch_metadata(self, url: str) -> dict[str, Any]:
        """Fetch video metadata without downloading."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-warnings",
            url,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise DownloadError("Metadata fetch timed out (30s)") from exc
        except subprocess.CalledProcessError as exc:
            raise DownloadError(f"Failed to fetch metadata: {exc.stderr.strip()}") from exc

        return json.loads(proc.stdout)  # type: ignore[no-any-return]

    def _download_video(self, url: str, output_path: Path) -> None:
        """Download best video+audio merged into mp4."""
        log_info("Downloading video...")

        # yt-dlp picks the best video+audio and merges into mp4
        cmd = [
            "yt-dlp",
            # Format: best video + best audio, prefer mp4 container
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
            "--merge-output-format",
            "mp4",
            "-o",
            str(output_path),
            "--newline",
            "--no-warnings",
            # Avoid post-processing issues
            "--no-playlist",
            "--no-part",
            url,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as exc:
            raise DownloadError("yt-dlp not found. Install it: brew install yt-dlp") from exc

        if proc.stdout is None:  # pragma: no cover
            proc.kill()
            proc.wait()
            msg = "stdout pipe was not opened"
            raise DownloadError(msg)

        try:
            with create_progress() as progress:
                task = progress.add_task("Downloading", total=100)

                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    # Parse yt-dlp progress lines like "[download]  45.2% of ..."
                    if "[download]" in line and "%" in line:
                        try:
                            pct_str = line.split("%")[0].split()[-1]
                            pct = float(pct_str)
                            progress.update(task, completed=pct)
                        except (ValueError, IndexError):
                            pass

                progress.update(task, completed=100)

            return_code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise

        if return_code != 0:
            raise DownloadError(f"yt-dlp exited with code {return_code}")

        if not output_path.exists():
            raise DownloadError("yt-dlp finished but output file not found")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        log_info(f"Downloaded {size_mb:.1f} MB")
