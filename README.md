# Fourier Analysis Problems

This repository contains solutions for three different problems involving Fourier Analysis, signal processing, and image processing.

## Prerequisites

Before running the code, you need to set up your environment.

### 1. Clone the Repository

Open your terminal or command prompt and run the following command to download the code to your local machine:

```bash
git clone <repository_url>
cd fourier
```

### 2. Set Up a Virtual Environment

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Once the virtual environment is activated, install the required Python libraries:

```bash
pip install pandas numpy matplotlib scipy opencv-python openpyxl
```

---

## Running the Problems

You can run the solution for each problem by executing the corresponding Python script from the root directory of the repository.

### Problem 1: Audio Cleaning
**Goal:** Remove a specific high-frequency "scanner beep" noise from an audio recording.

**How to run:**
```bash
python prob01/problem_1.py
```

### Problem 3: Image Phase Swapping
**Goal:** Demonstrate the importance of Phase vs Magnitude in images by swapping the phases of two images and reconstructing them.

**How to run:**
```bash
python prob03/problem_3.py
```

### Problem 4: Hybrid Image
**Goal:** To create an optical illusion that is based on low and high frequencies.

**How to run:**
```bash
python prob04/problem_4.py
```

