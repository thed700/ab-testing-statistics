# 📊 Statistical Hypothesis Testing & A/B Simulation
### 7-Day Hands-on Statistics Journey | Python & SciPy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.12%2B-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive data science framework for validating business hypotheses through rigorous statistical methods. This project bridges the gap between raw data and data-driven decision making.

[**Project Structure**](#-project-structure) • [**Curriculum**](#-topics-covered) • [**A/B Simulation**](#-final-project-simulation) • [**Installation**](#-installation)

---

## 📋 Executive Summary

**The Goal:** To master the application of frequentist statistics in real-world scenarios, specifically focusing on E-commerce A/B testing and hypothesis validation.

**Key Components:**
* **Statistical Tests:** T-tests, Chi-Square, and ANOVA.
* **Probabilistic Logic:** p-value intuition, Effect Size (Cohen’s d), and Confidence Intervals.
* **Simulation:** Building a synthetic A/B test environment to evaluate conversion rate uplifts.

---

## 🗂 Project Structure

```bash
ab_testing_project/
├── notebooks/
│   ├── 01_hypothesis_testing.ipynb  # Foundations & Logic
│   ├── 02_p_value.ipynb             # Distribution analysis
│   ├── 03_t_test.ipynb              # T-test applications
│   ├── 04_chi_square.ipynb          # Categorical testing
│   ├── 05_confidence_interval.ipynb # Uncertainty quantification
│   ├── 06_split_testing.ipynb       # Methodology
│   └── 07_ab_test_simulation.ipynb  # Final Case Study
├── data/
│   └── generate_data.py             # Synthetic data engine
├── src/
│   ├── stats_utils.py               # Reusable statistical functions
│   └── visualizer.py                # Publication-quality charts
├── requirements.txt
└── README.md
````

-----

## 📚 Topics Covered

| Day | Topic | Technical Focus | Notebook |
| :--- | :--- | :--- | :--- |
| **D15** | **Hypothesis Testing** | Null vs Alternative Hypotheses | `01_testing.ipynb` |
| **D16** | **p-value Intuition** | Type I & Type II Errors | `02_p_value.ipynb` |
| **D17** | **T-tests** | Independent & Paired Samples | `03_t_test.ipynb` |
| **D18** | **Chi-Square** | Categorical Independence | `04_chi_square.ipynb` |
| **D19** | **Confidence Intervals** | Error Margins & Bootstrapping | `05_interval.ipynb` |
| **D21** | **A/B Simulation** | Full conversion rate analysis | `07_simulation.ipynb` |

-----

## 🎯 Final Project Simulation

The capstone project simulates a high-stakes E-commerce decision: **Should we implement a new checkout flow?**

### Analysis Pipeline:

1.  **Metric Definition:** Primary KPI is Conversion Rate (CR).
2.  **Power Analysis:** Determining the required sample size for statistical significance.
3.  **Statistical Testing:** Applying Welch's T-test for continuous metrics.
4.  **Business Conclusion:** Evaluating if the observed lift justifies the engineering cost.

-----

## 🛠 Tech Stack

  * **Core:** Python 3.10+
  * **Analysis:** Pandas, NumPy
  * **Statistics:** SciPy Stats, Statsmodels
  * **Visualization:** Matplotlib, Seaborn (Dark-themed professional plots)

-----

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/thed700/ab-testing-statistics.git](https://github.com/thed700/ab-testing-statistics.git)
cd ab-testing-statistics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the journey
jupyter notebook
```

-----

## 🔬 Key Learnings

  * **Statistical Significance vs. Practical Significance:** A small p-value doesn't always mean a large business impact.
  * **Handling Skewed Data:** Applying transformations before testing.
  * **Standard Error:** Visualizing uncertainty through forest plots and error bars.

-----

## 👤 Author

**Akmal** — *Data Analyst & Statistical Learner*

  * **GitHub:** [@thed700](https://github.com/thed700)
  * **Telegram:** [@Eshituvchiy](https://www.google.com/search?q=https://t.me/Eshituvchiy)

-----

*License: MIT — Build, learn, and contribute.*

```
```
