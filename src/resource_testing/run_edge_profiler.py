import os
import torch
import torchaudio
from src.resource_testing.stt_edge_profiler import STTEdgeProfiler

# Model to test
MODEL_NAME = "StefanStefan/Wav2Vec-100-CSR-12M"

# Create a test directory
os.makedirs("edge_test_results", exist_ok=True)

# Test with your audio file
AUDIO_PATH = "DATA/M18_05_01.wav"

# If your file doesn't exist, create a test audio file
if not os.path.exists(AUDIO_PATH):
    print(f"Audio file not found: {AUDIO_PATH}")
    print("Creating a test file instead")

    test_path = "test_audio.wav"
    print(f"Creating test audio file: {test_path}")
    sample_rate = 16000
    duration = 10  # seconds

    # Generate a simple wave pattern with spoken-word-like characteristics
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # Mix several frequencies to simulate speech
    waveform = (
            0.3 * torch.sin(2 * 3.14159 * 150 * t) +  # Base tone
            0.2 * torch.sin(2 * 3.14159 * 450 * t) +  # Higher frequency
            0.1 * torch.sin(2 * 3.14159 * 1200 * t) +  # Consonant-like
            0.05 * torch.randn(t.shape)  # Noise
    ).unsqueeze(0)

    torchaudio.save(test_path, waveform, sample_rate)
    print(f"Test audio file created: {test_path}")
    AUDIO_PATH = test_path

# Create and run the profiler
print(f"Profiling {MODEL_NAME}...")
print("This will measure CPU, memory, and battery usage during transcription")

try:
    # Initialize profiler
    profiler = STTEdgeProfiler(model_name=MODEL_NAME, sampling_interval=0.1)

    # Run basic test first (whole audio file)
    print("\n==== Running basic inference test ====")
    summary1, metrics1, transcript1 = profiler.run_inference(
        AUDIO_PATH,
        num_repeats=3,
        stream_simulation=False
    )

    # Save results
    profiler.save_results(
        summary1,
        metrics1,
        AUDIO_PATH,
        transcript1,
        output_dir="edge_test_results"
    )

    # Run streaming test
    print("\n==== Running streaming simulation test ====")
    summary2, metrics2, transcript2 = profiler.run_inference(
        AUDIO_PATH,
        num_repeats=3,
        stream_simulation=True
    )

    # Save streaming results
    profiler.save_results(
        summary2,
        metrics2,
        AUDIO_PATH,
        transcript2,
        output_dir="edge_test_results"
    )

    # Visualize both results
    profiler.visualize_results(metrics1, summary1, output_dir="edge_test_results")
    profiler.visualize_results(metrics2, summary2, output_dir="edge_test_results")

    # Print combined summary
    print("\n==== SUMMARY RESULTS ====")
    print(f"Model: {MODEL_NAME}")
    print(f"Model size: 24 MB / 12M parameters")

    print("\nBasic Inference:")
    print(f"- Realtime factor: {summary1['realtime_factor']:.2f}x")
    print(f"- Memory usage: {summary1['max_memory_mb']:.1f} MB")
    print(f"- CPU usage: {summary1['avg_cpu_percent']:.1f}%")
    if 'battery_drain' in summary1:
        print(f"- Battery drain: {summary1['battery_drain']:.2f}%")
    print(f"- Edge suitability score: {summary1.get('edge_suitability_score', 'N/A')}/10")

    print("\nStreaming Inference:")
    print(f"- Realtime factor: {summary2['realtime_factor']:.2f}x")
    print(f"- Memory usage: {summary2['max_memory_mb']:.1f} MB")
    print(f"- CPU usage: {summary2['avg_cpu_percent']:.1f}%")
    if 'battery_drain' in summary2:
        print(f"- Battery drain: {summary2['battery_drain']:.2f}%")
    print(f"- Edge suitability score: {summary2.get('edge_suitability_score', 'N/A')}/10")

    print("\nEdge Device Suitability Assessment:")
    score = summary1.get('edge_suitability_score', 0)
    if score >= 8:
        print("✅ EXCELLENT: This model should work well on most edge devices")
    elif score >= 6:
        print("✅ GOOD: This model should work on mid-range edge devices")
    elif score >= 4:
        print("⚠️ FAIR: This model may work on high-end edge devices only")
    else:
        print("❌ POOR: This model is likely not suitable for edge deployment")

    # Results location
    print(f"\nDetailed results saved to: edge_test_results/")

except Exception as e:
    print(f"Error during profiling: {e}")
    import traceback

    traceback.print_exc()
