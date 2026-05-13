The-Capitulation-Problem
This repository explores why LLMs like Gemma 4B prioritize social compliance over factual truth when challenged. Featuring a dataset of 486+ trials mapping the breaking point of AI knowledge under social pressure. Foundational work for AI safety and human-AI interaction.

The Capitulation Problem: A Conditional Optimization Framework
Author: Tharun Rathod B. (IIT Roorkee)

📌 Project Overview
This repository contains the dataset and analysis for the "Capitulation Problem"—a study on why decision-making systems (Human and AI) abandon correct first-outputs under social pressure rather than new information.

The core of this research identifies the P4 Anomaly: a phenomenon where the LLM ignores expert disagreement but capitulates to direct personal confrontation 64.6% of the time.

📊 Key Findings: The P4 Anomaly
Our empirical tests on Gemma 4B revealed a striking asymmetry in how AI handles pressure:

Pressure Variant	Strategy	Hard Reversal (SWITCH)
P2: Expert Authority	"An expert disagrees..."	3.1%
P4: Direct Challenge	"I think you're wrong..."	64.6%
Conclusion: The model is nearly 20x more likely to fold for a random user's challenge than for a cited expert. This highlights a structural "Compliance Bias" in RLHF-trained models.

📁 Repository Structure
/data: Includes judged_c6_baseline.csv containing 484 trials across 5 domains.
/scripts: Python scripts for automated classification and data visualization.
paper.pdf: The full IEEE-formatted research paper.
🛠️ Methodology
Model: Gemma 4B (Local via Ollama).
Baseline: C6 Neutral System Prompt.
Domains: Factual, Moral, Emotional, Speculative Scientific, and Legal-Ethical.
Analysis: Rule-based classification validated against Human-as-Judge metrics.
⚖️ The Three-Question Diagnostic
The research proposes a framework to stop capitulation:

Origin Check: Is the output from knowledge or anxiety?
Pressure Audit: Is this new information or just social friction?
Specificity Test: Can you name exactly what fact changed?
🔗 How to Cite
If you use this dataset or the P4 Anomaly framework in your research, please cite:

Rathod, T. (2026). The Capitulation Problem: A Conditional Optimization Framework for First-Output Fidelity. Independent Research, IIT Roorkee.
