# Acoustic Physical Neural Network (Acoustic PNN) - Example Experiment Pipeline

Welcome to the example repository for the Acoustic PNN project. This repository contains a fully structured experimental pipeline and the corresponding phase manuscript, demonstrating a hardware-in-the-loop physics-informed learning system. 

## Executive Summary
Traditional digital computing hardware faces severe energy and computational bottlenecks in the AI era. This project explores the physical neural network (PNN) paradigm by utilizing acoustic wave propagation as the physical forward pass, optimized via the **Physical Local Learning (phyLL)** algorithm instead of traditional Backpropagation (BP).

**Key Hardware & Physical Setup:**
* **Custom Acoustic Cavity:** A hermetically sealed 10mm-thick acrylic cavity used to map discrete digital inputs into an infinite-dimensional continuous acoustic state space.
* **Transducer Array:** 4 independent speakers (102mm primary aperture) act as the excitation source, and 4 microphones capture the physically modulated acoustic responses.
* **Anechoic Environment:** The entire hardware system is isolated within a custom anechoic chamber to suppress external environmental noise.

**Algorithm & Engineering Optimization:**
To bridge the gap between unpredictable physical systems and digital networks, robust signal processing and system tuning mechanisms were implemented:
* **Pre-experiment Frequency Sweeping:** Automatically scans (200-1200Hz) to bypass extreme resonance zones and acoustic nodes, selecting the 5 most stable baseline frequencies.
* **Steady-State Energy Truncation:** Dynamically calculates the smoothed energy envelope, skipping the 150ms initial transient phase and extracting a stable 200ms valid signal window.
* **High-Precision Amplitude Extraction:** Utilizes a Flattop window combined with **Parabolic Sub-bin Interpolation** to correct the "picket-fence effect" in FFT, obtaining high-fidelity physical resonance amplitudes.
* **94th-Percentile Normalization:** Replaces global maximum normalization with a dynamic thresholding strategy, effectively suppressing acoustic extremes and releasing the numerical distribution space for valid micro-signals.

**Performance:**
Tested on a 7-class Vowel Dataset, the optimized acoustic computing platform achieved a **96% classification accuracy** (on both training and test sets), firmly validating the feasibility of executing high-dimensional feature mapping in acoustic PNNs.

---

## Repository Structure

The repository is divided into two main sections to separate documentation from the executable pipeline:

* **`Docs/`**: Contains the detailed phase manuscript (in Chinese). 
    * *Note: For comprehensive theoretical derivations, detailed hardware schematics, and experimental data analysis, please refer to the attached PDF report.*
* **`Experiment_Pipeline/`**: The complete code and data workspace for this specific example run. All scripts and raw data (including the original `Vowel_dataset.xlsx`) are kept in a flat directory to ensure absolute path stability.

---

## Pipeline Workflow

The experimental pipeline in `Experiment_Pipeline/` illustrates the complete "Digital-Physical-Digital" data flow:

1. **Data Encoding & Audio Generation (`1input_generator.py` & `2audio_generator.py`)**
   * Parses the 12-dimensional features and 7-dimensional One-hot labels into a unified 20-dimensional input vector.
   * Maps the vector to 5 baseline frequencies and generates multi-channel acoustic excitation signals (48 kHz) with fade-in/fade-out sinusoidal envelopes to prevent transient step noise.
2. **Physical Forward Pass (`3audio_player.py`)**
   * Interfaces with the hardware to inject the encoded audio into the 10mm acrylic cavity, allowing physical reflection and interference to act as the computational transformation.
3. **Signal Analysis & Normalization (`4audio_analyzer.py` & `7phys_normalizer.py`)**
   * Executes the FFT and Parabolic Sub-bin Interpolation to extract 20-dimensional physical embeddings from the 4 microphone channels.
   * Applies the 94th-percentile truncation to normalize the physical state data.
4. **Model Training (`5phyll_trainer.py`)**
   * Evaluates the physical outputs against reference targets using the Goodness metric (cosine similarity).
   * Updates the digital Readout layer weights directly via the phyLL algorithm, fully bypassing global backpropagation.
5. **Diagnostics & Visualization (`6visualizer.py`, `data_diagnostics.py`, `data_inspector.py`)**

---

## Dataset & Raw Acoustic Recordings Download

Due to GitHub's file size limits for high-resolution, multi-channel audio, the complete set of raw acoustic recordings and the original dataset have been hosted externally. 

**[Download the Complete Experimental Data Here] https://drive.google.com/file/d/16aGko-q0xxtIWQK3i7W1ZaE04hw0dGia/view?usp=drive_link
**

**Setup Instructions:**
To ensure the pipeline scripts run flawlessly without modifying any relative paths, please download the `.zip` file from the link above and extract all its contents (e.g., `Sweep_1/`, `train_pos.../` , `train_neg.../`) directly into the `Experiment_Pipeline/` directory.

---