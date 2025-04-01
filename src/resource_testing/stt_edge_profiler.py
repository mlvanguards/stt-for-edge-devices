import os
import time
import psutil
import threading
import numpy as np
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

from src.config.settings import settings
from src.utils.audio.audio_process import AudioProcessor


class STTEdgeProfiler:
    """
    Edge device profiler for speech-to-text models.
    Measures performance metrics like CPU usage, memory usage, and inference time.
    Uses the centralized AudioProcessor for all audio operations.
    """

    def __init__(self, model_name: str, sampling_interval: Optional[float] = None,
                 audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize the profiler for a Wav2Vec2 Speech-to-Text model with edge performance focus.

        Args:
            model_name: Huggingface STT model name (e.g., 'StefanStefan/Wav2Vec-100-CSR-12M')
            sampling_interval: How often to sample resource usage (seconds)
            audio_processor: Optional audio processor instance
        """
        self.model_name = model_name
        # Use settings value if not provided
        self.sampling_interval = sampling_interval or settings.testing.TESTING_SAMPLING_INTERVAL
        self.metrics = []
        self.monitoring = False
        self.process = psutil.Process(os.getpid())
        self.audio_processor = audio_processor or AudioProcessor()

        # Check if running on battery (only works on laptops)
        self.battery_available = hasattr(psutil, "sensors_battery") and psutil.sensors_battery() is not None

        # Store device information - with Mac MPS support
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA GPU")
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
            print("Using Apple Silicon MPS")
        else:
            self.device = settings.testing.TESTING_DEFAULT_DEVICE
            print(f"Using {self.device}")

        print(f"Loading STT model {model_name} on {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            print("STT model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _monitor_resources(self) -> None:
        """Background thread to monitor resource usage"""
        while self.monitoring:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()

            # Get GPU info if available
            gpu_memory_used = 0
            if self.device == "cuda":
                gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            elif self.device == "mps":  # For Apple Silicon GPUs
                gpu_memory_used = memory_info.rss / (1024 * 1024)

            # Get battery info if available (works on MacBooks)
            battery_percent = None
            if self.battery_available:
                battery = psutil.sensors_battery()
                if battery:
                    battery_percent = battery.percent

            # Get CPU temperature if available
            cpu_temp = None
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps and "coretemp" in temps:
                    cpu_temp = sum(temp.current for temp in temps["coretemp"]) / len(temps["coretemp"])

            self.metrics.append({
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / (1024 * 1024),  # RSS in MB
                'memory_vms_mb': memory_info.vms / (1024 * 1024),  # VMS in MB
                'gpu_memory_mb': gpu_memory_used,
                'battery_percent': battery_percent,
                'cpu_temp': cpu_temp
            })

            time.sleep(self.sampling_interval)

    def start_monitoring(self) -> 'STTEdgeProfiler':
        """
        Start the resource monitoring thread.

        Returns:
            Self for method chaining
        """
        self.monitoring = True
        self.metrics.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return self

    def stop_monitoring(self) -> 'STTEdgeProfiler':
        """
        Stop the resource monitoring thread.

        Returns:
            Self for method chaining
        """
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        return self

    def run_inference(self, audio_path: str, num_repeats: Optional[int] = None, stream_simulation: bool = False) -> \
    Tuple[Dict, pd.DataFrame, str]:
        """
        Run STT inference on the given audio file and monitor resource usage.

        Args:
            audio_path: Path to audio file
            num_repeats: Number of times to repeat inference for better measurements
            stream_simulation: Simulate streaming by processing chunks

        Returns:
            Tuple of (summary statistics, detailed metrics DataFrame, transcribed text)
        """
        # Use settings value if not provided
        num_repeats = num_repeats or settings.testing.TESTING_DEFAULT_NUM_REPEATS

        # Load and preprocess audio using AudioProcessor
        print(f"Loading audio file: {audio_path}")
        waveform, sample_rate = self.audio_processor.load_audio(audio_path)

        # Convert to numpy for compatibility with transformers
        audio_array = waveform.squeeze().numpy()
        audio_length_seconds = len(audio_array) / sample_rate
        print(f"Audio length: {audio_length_seconds:.2f} seconds")

        # Start monitoring
        self.start_monitoring()

        # Record inference times and transcriptions
        inference_times = []
        transcription = None

        try:
            # Simulate streaming by processing chunks (if requested)
            if stream_simulation and audio_length_seconds > 3.0:
                print("Simulating streaming audio processing...")
                chunk_size = int(1.0 * sample_rate)  # 1 second chunks
                chunk_results = []

                for i in range(num_repeats):
                    print(f"Running streaming inference {i + 1}/{num_repeats}...")

                    # Clear cuda/mps cache if available
                    if self.device in ["cuda", "mps"]:
                        torch.cuda.empty_cache() if self.device == "cuda" else None

                    total_time = 0
                    for start_idx in range(0, len(audio_array), chunk_size):
                        end_idx = min(start_idx + chunk_size, len(audio_array))
                        chunk = audio_array[start_idx:end_idx]

                        # Process chunk - modified for Wav2Vec2
                        chunk_inputs = self.processor(chunk, sampling_rate=settings.audio.AUDIO_SAMPLE_RATE,
                                                      return_tensors="pt")
                        chunk_inputs = {k: v.to(self.device) for k, v in chunk_inputs.items()}

                        # Time the inference
                        start_time = time.time()
                        with torch.no_grad():
                            outputs = self.model(**chunk_inputs).logits
                            predicted_ids = torch.argmax(outputs, dim=-1)

                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        elif self.device == "mps":
                            torch.mps.synchronize()

                        end_time = time.time()
                        total_time += (end_time - start_time)

                        # Decode the last chunk result
                        if i == num_repeats - 1:
                            chunk_text = self.processor.batch_decode(predicted_ids)
                            chunk_results.append(chunk_text[0])

                    inference_times.append(total_time)
                    print(f"Total streaming inference time: {total_time:.4f} seconds")

                # Combine chunk results for final transcription
                if chunk_results:
                    transcription = " ".join(chunk_results)
            else:
                # Standard whole-file processing
                for i in range(num_repeats):
                    print(f"Running inference {i + 1}/{num_repeats}...")

                    # Clear cuda/mps cache if available
                    if self.device in ["cuda", "mps"]:
                        torch.cuda.empty_cache() if self.device == "cuda" else None

                    # Prepare inputs for Wav2Vec2
                    inputs = self.processor(audio_array, sampling_rate=settings.audio.AUDIO_SAMPLE_RATE,
                                            return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Measure inference time
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = self.model(**inputs).logits
                        predicted_ids = torch.argmax(outputs, dim=-1)

                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    elif self.device == "mps":
                        torch.mps.synchronize()

                    end_time = time.time()

                    # Save inference time
                    inference_time = end_time - start_time
                    inference_times.append(inference_time)
                    print(f"Inference time: {inference_time:.4f} seconds")

                    # Only decode the last one to avoid overhead in timing measurements
                    if i == num_repeats - 1:
                        transcription = self.processor.batch_decode(predicted_ids)[0]
        finally:
            # Stop monitoring
            self.stop_monitoring()

        # Calculate results
        df = pd.DataFrame(self.metrics)

        # Edge-focused metrics
        summary = {
            'model_name': self.model_name,
            'device': self.device,
            'audio_length_seconds': audio_length_seconds,
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'realtime_factor': np.mean(inference_times) / audio_length_seconds,
            'max_cpu_percent': df['cpu_percent'].max() if not df.empty else 0,
            'avg_cpu_percent': df['cpu_percent'].mean() if not df.empty else 0,
            'max_memory_mb': df['memory_rss_mb'].max() if not df.empty else 0,
            'avg_memory_mb': df['memory_rss_mb'].mean() if not df.empty else 0,
            'memory_footprint_mb': df['memory_rss_mb'].max() - df['memory_rss_mb'].iloc[0] if not df.empty else 0,
            'model_size_mb': 24,  # Hardcoded for StefanStefan/Wav2Vec-100-CSR-12M
            'streaming_mode': stream_simulation
        }

        # Add GPU metrics if available
        if self.device in ["cuda", "mps"] and not df.empty and 'gpu_memory_mb' in df:
            summary['max_gpu_memory_mb'] = df['gpu_memory_mb'].max()
            summary['avg_gpu_memory_mb'] = df['gpu_memory_mb'].mean()

        # Add battery metrics if available
        if self.battery_available and not df.empty and 'battery_percent' in df and df['battery_percent'].notna().any():
            # Calculate rate of battery drain per minute of audio
            initial_battery = df['battery_percent'].iloc[0]
            final_battery = df['battery_percent'].iloc[-1]
            battery_drain = initial_battery - final_battery

            if battery_drain > 0:
                # Normalize to 1 hour of audio
                battery_drain_per_hour_audio = (battery_drain / audio_length_seconds) * 3600
                summary['battery_drain'] = battery_drain
                summary['battery_drain_per_hour_audio'] = battery_drain_per_hour_audio

                # Estimate total transcription time possible on battery
                if battery_drain_per_hour_audio > 0:
                    estimated_hours = initial_battery / battery_drain_per_hour_audio
                    summary['estimated_transcription_hours'] = estimated_hours

        # Estimate edge device suitability score (0-10 scale)
        edge_score = 10.0

        # Penalize for high memory usage
        memory_threshold = settings.testing.TESTING_EDGE_MEMORY_THRESHOLD_MB
        if summary['max_memory_mb'] > memory_threshold:
            edge_score -= min(5, (summary['max_memory_mb'] - memory_threshold) / 1000)

        # Penalize for slow processing (realtime factor > 1.0 is bad)
        if summary['realtime_factor'] > 1.0:
            edge_score -= min(5, (summary['realtime_factor'] - 1.0) * 5)

        # Penalize for high CPU usage
        cpu_threshold = settings.testing.TESTING_EDGE_CPU_THRESHOLD_PERCENT
        if summary['avg_cpu_percent'] > cpu_threshold:
            edge_score -= min(3, (summary['avg_cpu_percent'] - cpu_threshold) / 20)

        # Clamp score between 0-10
        edge_score = max(0, min(10, edge_score))
        summary['edge_suitability_score'] = edge_score

        return summary, df, transcription

    def save_results(self,
                     summary: Dict[str, Any],
                     df: pd.DataFrame,
                     audio_path: str,
                     transcription: Optional[str] = None,
                     output_dir: str = "./stt_profiling_results") -> Tuple[str, str]:
        """
        Save the profiling results.

        Args:
            summary: Summary metrics dictionary
            df: DataFrame with detailed metrics
            audio_path: Path to the audio file used
            transcription: Optional transcription text
            output_dir: Directory to save results

        Returns:
            Tuple of (summary_file_path, details_file_path)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.model_name.replace('/', '_')
        audio_name = os.path.basename(audio_path).split('.')[0]

        # Save summary as JSON
        summary_file = f"{output_dir}/{model_name_safe}_{audio_name}_{timestamp}_summary.json"
        pd.Series(summary).to_json(summary_file)

        # Save detailed metrics as CSV
        details_file = f"{output_dir}/{model_name_safe}_{audio_name}_{timestamp}_details.csv"
        df.to_csv(details_file, index=False)

        # Save transcription if available
        if transcription:
            transcript_file = f"{output_dir}/{model_name_safe}_{audio_name}_{timestamp}_transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"Transcription saved to {transcript_file}")

        print(f"Results saved to {summary_file} and {details_file}")
        return summary_file, details_file

    def visualize_results(self, df: pd.DataFrame, summary: Dict[str, Any],
                          output_dir: str = "./stt_profiling_results") -> None:
        """
        Generate visualization of resource usage.

        Args:
            df: DataFrame with detailed metrics
            summary: Summary metrics dictionary
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt

            if df.empty:
                print("No data to visualize")
                return

            model_name_safe = self.model_name.replace('/', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create plots directory
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(12, 12))

            # Plot CPU usage
            plt.subplot(4, 1, 1)
            plt.plot(range(len(df)), df['cpu_percent'], label='CPU %', color='green')
            plt.axhline(y=settings.testing.TESTING_EDGE_CPU_THRESHOLD_PERCENT, color='r', linestyle='--', alpha=0.3,
                        label='Edge device threshold')
            plt.title(f'CPU Usage - {self.model_name}')
            plt.ylabel('CPU %')
            plt.grid(True)
            plt.legend()

            # Plot Memory usage
            plt.subplot(4, 1, 2)
            plt.plot(range(len(df)), df['memory_rss_mb'], label='Memory (MB)', color='blue')
            plt.axhline(y=settings.testing.TESTING_EDGE_MEMORY_THRESHOLD_MB, color='r', linestyle='--', alpha=0.3,
                        label='2GB Edge threshold')
            plt.title('Memory Usage')
            plt.ylabel('Memory (MB)')
            plt.grid(True)
            plt.legend()

            # Plot GPU Memory if available
            if self.device in ["cuda", "mps"] and 'gpu_memory_mb' in df.columns:
                plt.subplot(4, 1, 3)
                plt.plot(range(len(df)), df['gpu_memory_mb'], label='GPU Memory (MB)', color='purple')
                plt.title(f'GPU Memory Usage ({self.device})')
                plt.ylabel('GPU Memory (MB)')
                plt.grid(True)
                plt.legend()

            # Plot Battery if available
            if 'battery_percent' in df.columns and df['battery_percent'].notna().any():
                plt.subplot(4, 1, 4)
                plt.plot(range(len(df)), df['battery_percent'], label='Battery %', color='orange')
                plt.title('Battery Level')
                plt.xlabel('Sample')
                plt.ylabel('Battery %')
                plt.grid(True)
                plt.legend()

            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name_safe}_profile_{timestamp}.png")
            print(f"Visualization saved to {output_dir}/{model_name_safe}_profile_{timestamp}.png")
            plt.close()

            # Create a summary image for edge device estimation
            plt.figure(figsize=(8, 6))
            plt.bar(['Memory (GB)', 'RT Factor', 'CPU %/100', 'Edge Score/10'],
                    [summary['max_memory_mb'] / 1000, summary['realtime_factor'],
                     summary['avg_cpu_percent'] / 100, summary['edge_suitability_score'] / 10],
                    color=['blue', 'green', 'orange', 'purple'])
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            plt.title('Edge Device Suitability Metrics')
            plt.savefig(f"{output_dir}/{model_name_safe}_edge_estimate_{timestamp}.png")
            print(f"Edge estimate visualization saved to {output_dir}/{model_name_safe}_edge_estimate_{timestamp}.png")
            plt.close()
        except ImportError:
            print("Matplotlib not installed. Skipping plot generation.")
