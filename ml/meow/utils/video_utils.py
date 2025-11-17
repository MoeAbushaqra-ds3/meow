from typing import Dict, Union, List, Tuple, Optional

from .eval_utils import eval_expr
from .file_utils import create_temporary_file_name_with_extension


from tempfile import NamedTemporaryFile
import ffmpeg
from moviepy.video.VideoClip import VideoClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from ..logger import setup_logger
import subprocess

import os
import tempfile
from pathlib import Path

logger = setup_logger(__name__)


def get_video_info(video_path) -> Dict[str, Union[int, float, str]]:
    """
    Return JSON representing video info:
        file path
        frame width in pixels
        frame high in pixels
        frame rate as FPS
        duration in seconds
    """
    probe = ffmpeg.probe(video_path)
    file_path = str(probe['format']['filename'])
    file_type = os.path.splitext(file_path)[1].lower().replace(".", "")
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    duration_str = probe['format'].get('duration')
    duration = float(duration_str) if duration_str is not None else 0.0

    if video_stream is None:
        logger.warning("No video stream detected in %s, falling back to OpenCV metadata.", video_path)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file {video_path} for metadata inspection")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()

        frame_rate = round(fps) if fps and fps > 0 else 0
        if duration == 0.0 and fps and fps > 0:
            duration = float(frame_count / fps)
    else:
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        frame_rate = round(eval_expr(video_stream['avg_frame_rate']))

    if frame_rate == 0:
        logger.warning("Unable to determine FPS for %s, defaulting to 30 FPS.", video_path)
        frame_rate = 30

    return {
        "file_path": file_path,
        "file_type": file_type,
        "frame_height": height,
        "frame_width": width,
        "frame_rate": frame_rate,
        "duration": duration
    }


def get_num_frames(video_path: str) -> int:
    video_streams = [s for s in ffmpeg.probe(video_path)["streams"] if s["codec_type"] == "video"]
    assert len(video_streams) == 1
    return int(video_streams[0]["nb_frames"])

def concatenate_video_clips(video_file_paths) -> VideoClip:
    clips = [VideoFileClip(file) for file in video_file_paths]
    final_clip = concatenate_videoclips(clips)
    return final_clip


def ffmpeg_concatenate_video_clips(video_file_paths: List[str], output_path: Optional[str] = None, temp_dir: Optional[str] = None, file_type: Optional[str] = None):
    """Concatenate video clips and optionally transform FPS.
    
    Args:
        video_file_paths: List of video file paths to concatenate
        output_path: Path for the output video
        output_fps: Target FPS for output video. If None, keeps original FPS.
    """

    if output_path is None:
        output_video_path = create_temporary_file_name_with_extension(temp_dir, file_type)
    else:
        output_video_path = output_path

    logger.info(f"Concatenating videos to {output_video_path}")

    merged_video_list_file = NamedTemporaryFile(suffix=".txt")

    with open(merged_video_list_file.name, 'w') as f:
        for video in video_file_paths:
            escaped_path = f"'{video.replace(chr(92), chr(92)*2).replace(chr(39), chr(92)+chr(39))}'"
            f.write(f"file {escaped_path}\n")

    merged_video_list_file.seek(0)
    
    ffmpeg.input(merged_video_list_file.name, format='concat', safe=0).output(
        output_video_path, 
        codec='copy'
    ).global_args('-nostdin').run()

    return output_video_path


def get_last_frame(capture, duration):
    # Dirty hack to get to last 2 seconds to avoid reading to whole video file
    capture.set(cv2.CAP_PROP_POS_MSEC, (duration - 2) * 1000)
    last_frame = None
    while True:
        ret, tmp_frame = capture.read()
        if not ret:
            break
        last_frame = tmp_frame

    success = last_frame is not None
    return success, last_frame


def run_ffmpeg_fps_transform_with_progress(input_path: str, output_path: str, output_fps: int, vcodec: str, hw_accel: Optional[str] = None):
    try:
        if hw_accel is not None:
            logger.info(f"Transforming video FPS to {output_fps} using {hw_accel} acceleration")
            stream = (
                ffmpeg
                .input(input_path)
                .output(output_path,
                        r=output_fps,
                        vcodec=vcodec,
                        acodec='copy',
                        crf=24,
                )
                .global_args('-hwaccel', hw_accel)
            )
        else:
            logger.info(f"Transforming video FPS to {output_fps} using software encoding")
            stream = (
                ffmpeg
                .input(input_path)
                .output(output_path,
                        r=output_fps,
                        vcodec=vcodec,
                        acodec='copy',
                        crf=24,
                )
            )

        stream.run()
        logger.info(f"Successfully transformed FPS using {vcodec}")
        return output_path

    except ffmpeg.Error as e:
        logger.debug(f"{vcodec} acceleration failed: {str(e)}")
        raise


def transform_video_fps(input_path: str, output_fps: int, output_path: Optional[str] = None, temp_dir: Optional[str] = None, file_type: Optional[str] = None) -> str:
    """Transform video FPS using hardware acceleration when available.
    Falls back to software encoding if hardware acceleration fails."""
    if output_path is None:
        output_path = create_temporary_file_name_with_extension(temp_dir, file_type)
    
    logger.info(f"Transforming video FPS to {output_fps} using hardware acceleration")
    
    # Check available hardware encoders
    hw_encoders = get_available_hw_encoders()

    if hw_encoders['nvidia']:
        return run_ffmpeg_fps_transform_with_progress(input_path, output_path, output_fps, 'h264_nvenc', 'cuda')
    
    if hw_encoders['quicksync']:
        return run_ffmpeg_fps_transform_with_progress(input_path, output_path, output_fps, 'h264_qsv', 'qsv')
    
    if hw_encoders['vaapi']:
        return run_ffmpeg_fps_transform_with_progress(input_path, output_path, output_fps, 'h264_vaapi', 'vaapi')
    
    logger.info("No working hardware acceleration found, falling back to software encoding")
    return run_ffmpeg_fps_transform_with_progress(input_path, output_path, output_fps, 'libx265', None)


def cut_subclip_with_ffmpeg(input_path: str, output_path: str, start_time: float, end_time: float, output_args: dict):
    """Helper function to cut a single video with ffmpeg."""
    try:
        stream = (
            ffmpeg
            .input(input_path, ss=start_time)
            .output(output_path,
                   t=end_time-start_time,
                   **output_args)
            .overwrite_output()
        )
        stream.run()
    except ffmpeg.Error as e:
        logger.error(f"Error cutting video {input_path}: {str(e)}")
        raise

def cut_clips_with_ffmpeg(temp_dir: str, file_type: str, start_time: float, end_time: float, left_video_path: str, right_video_path: str) -> Tuple[str, str]:
    """
    Cut video clips with ffmpeg based on start and end time.
    Returns paths to both cut video files.
    """
    preprocessed_video_left_path = create_temporary_file_name_with_extension(temp_dir, file_type)
    preprocessed_video_right_path = create_temporary_file_name_with_extension(temp_dir, file_type)
    ffmpeg_extract_subclip(left_video_path, start_time, end_time, preprocessed_video_left_path)
    ffmpeg_extract_subclip(right_video_path, start_time, end_time,
                            preprocessed_video_right_path)
    
    return preprocessed_video_left_path, preprocessed_video_right_path


def ffmpeg_extract_subclip(filename: str, t1: float, t2: float, target_name: str = None):
    """ Makes a new video file playing video file ``filename`` between
    the times ``t1`` and ``t2``. t1 and t2 are in seconds."""
    name, ext = os.path.splitext(filename)
    if not target_name:
        T1, T2 = [int(1000 * t) for t in [t1, t2]]
        target_name = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [
        get_setting("FFMPEG_BINARY"),
        "-y",
        "-ss", "%0.2f" % t1,
        "-i", filename,
        "-t", "%0.2f" % (t2 - t1),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        target_name
    ]

    subprocess_call(cmd)


def extract_subclip(input_video_path, start_time, end_time) -> VideoFileClip:
    video = VideoFileClip(input_video_path)
    subclip: VideoFileClip = video.subclip(start_time, end_time)
    return subclip


def transcode_video(video_path: str, temp_dir: Optional[str] = None, output_path: Optional[str] = None,
                    video_codec: str = 'libx264') -> str:
    """Transcode a video to a standard container using libx264."""
    base_dir = temp_dir or os.path.dirname(video_path) or tempfile.gettempdir()
    os.makedirs(base_dir, exist_ok=True)

    if output_path is None:
        output_path = create_temporary_file_name_with_extension(base_dir, "mp4")
    else:
        os.makedirs(os.path.dirname(output_path) or base_dir, exist_ok=True)

    logger.info("Transcoding video %s to %s (%s)", video_path, output_path, video_codec)
    (
        ffmpeg
        .input(video_path)
        .output(
            output_path,
            vcodec=video_codec,
            pix_fmt='yuv420p',
            preset='fast',
            crf=20,
            an=None,
            movflags='+faststart'
        )
        .global_args('-nostdin')
        .run(overwrite_output=True)
    )
    return output_path


def merge_video_and_audio(video_path: str, audio_path: str, output_path: str, output_fps: int = None, overwrite: bool = False, temp_dir: Optional[str] = None):
    """Merge video and audio using FFMPEG stream copy or AAC as fallback.
    
    Args:
        video_path: Path to input video
        audio_path: Path to input audio
        output_path: Path to output file
        output_fps: Output framerate (optional)
    """
    duration = None
    try:
        video_info = get_video_info(video_path)
        duration = video_info['duration']
    except Exception as exc:
        logger.warning("Unable to read metadata from %s, continuing merge without duration clamp: %s",
                       video_path, exc)

    def _run_mux(src_video: str):
        input_video = ffmpeg.input(src_video)
        input_audio = ffmpeg.input(audio_path)

        output_kwargs = {
            "vcodec": "copy",
            "acodec": "aac"
        }
        if output_fps:
            output_kwargs["r"] = output_fps
        if duration:
            output_kwargs["t"] = duration

        (
            ffmpeg
            .output(
                input_video.video,
                input_audio.audio,
                output_path,
                shortest=None,
                **output_kwargs
            )
            .global_args('-nostdin')
            .run(overwrite_output=overwrite)
        )

    try:
        _run_mux(video_path)
    except ffmpeg.Error as mux_error:
        error_message = mux_error.stderr.decode('utf-8', errors='ignore') if mux_error.stderr else str(mux_error)
        logger.warning("Direct video/audio mux failed: %s", error_message)
        sanitized_path = None
        try:
            sanitized_path = transcode_video(video_path, temp_dir=temp_dir)
            _run_mux(sanitized_path)
        finally:
            if sanitized_path and os.path.exists(sanitized_path):
                try:
                    os.remove(sanitized_path)
                except OSError:
                    logger.debug("Could not remove temporary sanitized video %s", sanitized_path)


def ensure_h264_video(video_path: str, output_file_type: str, temp_dir: Optional[str] = None, require_transcode: bool = True) -> str:
    """Ensure stitched output is a standard H.264 MP4/MOV to keep downstream ffmpeg happy."""
    desired_ext = (output_file_type or "").lower()
    if desired_ext not in ("mp4", "mov"):
        return video_path

    needs_transcode = False
    codec_name = None
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            needs_transcode = True
        else:
            codec_name = video_stream.get("codec_name", "").lower()
            if codec_name not in {"h264", "hevc", "h265", "mp4v", "mpeg4"}:
                needs_transcode = True
    except ffmpeg.Error as exc:
        logger.warning("ffprobe failed for %s: %s. Re-encoding.", video_path, exc)
        needs_transcode = True

    current_ext = Path(video_path).suffix.lower().replace(".", "")
    if not needs_transcode and current_ext == desired_ext:
        return video_path

    if not needs_transcode and not require_transcode:
        return video_path

    base_dir = temp_dir or os.path.dirname(video_path) or tempfile.gettempdir()
    os.makedirs(base_dir, exist_ok=True)
    target_path = video_path if current_ext == desired_ext else create_temporary_file_name_with_extension(
        base_dir,
        desired_ext
    )
    return transcode_video(video_path, temp_dir=temp_dir, output_path=target_path)


def get_available_hw_encoders() -> dict:
    """Check available hardware encoders using ffmpeg."""
    logger.debug("Checking available hardware encoders")
    encoders = {
        'nvidia': False,
        'quicksync': False,
        'vaapi': False
    }

    try:
        result = subprocess.run(    
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True
        )

        output = result.stdout.lower()

        if 'h264_nvenc' in output:
            encoders['nvidia'] = True
            logger.debug("NVIDIA NVENC encoder available")
            
        if 'h264_qsv' in output:
            encoders['quicksync'] = True
            logger.debug("Intel QuickSync encoder available")
            
        if 'h264_vaapi' in output:
            encoders['vaapi'] = True
            logger.debug("VAAPI encoder available")
        
        
    except Exception as e:
        logger.error(f"Error checking hardware encoders: {str(e)}")

    return encoders


if __name__ == "__main__":
    print(get_available_hw_encoders())
