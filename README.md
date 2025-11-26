Markdown# Thermal Process Control System Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Project Overview

This project is a comprehensive simulation and analysis of a thermal process control system using Python. It demonstrates the full control engineering workflow, including mathematical modeling, system identification from noisy data, signal processing, and PID controller design.

The script simulates a first-order thermal system with transport delay and evaluates its performance in both the time and frequency domains.

## 🚀 Key Features

* **Process Modeling:** Implementation of a First-Order Plus Dead Time (FOPDT) transfer function:
    $$G(s) = \frac{3.5}{22s + 1}$$
* **Transient & Frequency Analysis:** Generation of Step Response and Bode Plots (Magnitude/Phase).
* **System Identification:** Simulation of experimental data with Gaussian noise, followed by parameter estimation to recover system constants ($K$, $\tau$).
* **Model Validation:** Statistical residual analysis (Normality, Homoscedasticity tests).
* **Signal Processing:** Comparison of Moving Average vs. Butterworth Low-Pass filtering for noise reduction (SNR improvement).
* **PID Control:** Design of a closed-loop PID controller with performance evaluation (Rise time, Overshoot, Settling time).
* **Performance Metrics:** Calculation of IAE, ISE, ITAE, and ITSE indices.

## 📊 Visualizations

The script generates 6 professional-grade plots using `matplotlib`:

1.  **Step Response:** Open-loop system reaction to a step input.
2.  **Bode Plot:** Frequency response analysis (Gain and Phase margins).
3.  **System Identification:** Comparison of True, Measured (Noisy), and Identified models.
4.  **Residual Analysis:** Statistical validation of the identified model.
5.  **Signal Filtering:** Noise reduction comparison (Raw vs. Filtered).
6.  **PID Control:** Closed-loop response against a setpoint.

## 🛠️ Installation & Requirements

Ensure you have Python installed. The project relies on the following scientific computing libraries:

```bash
pip install numpy matplotlib scipy pandas
💻 UsageSimply run the main script in your terminal:Bashpython thermal_control_analysis.py
(Replace thermal_control_analysis.py with your actual filename)📈 Example OutputUpon execution, the script calculates rigorous performance metrics. Below is an example of the generated summary:ParameterValueSpecificationStatusModel Fit (R²)0.99997> 0.99✅ PASSRMSE0.0421°C< 0.05°C✅ PASSRise Time12.45s< 30s✅ PASSOvershoot4.32%< 10%✅ PASSSteady-State Error0.00%< 2%✅ PASS🧠 TheoryThe core thermal system is modeled using the transfer function:$$G(s) = \frac{K}{\tau s + 1} e^{-\theta s}$$Where:$K$ = Process Gain (3.5 °C/%)$\tau$ = Time Constant (22.0 s)$\theta$ = Transport Delay (2.0 s)🤝 ContributingContributions are welcome! Please feel free to submit a Pull Request.📄 LicenseThis project is open-source and available under the MIT License.
***

### **How to add this to your project**

1.  Create a new file in your project folder named `README.md` (no file extension other than .md).
2.  Paste the content above into that file.
3.  Run the following commands in your terminal to update GitHub:

```bash
git add README.md
git commit -m "Add project documentation"
git push
