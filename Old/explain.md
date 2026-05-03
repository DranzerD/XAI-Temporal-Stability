Perfect. Use this exactly as your viva script.

**Slide 1: Problem Statement**
On slide:
1. Credit models are monitored for AUC decline.
2. But explanations can drift before AUC drops.
3. In finance, unstable reasoning is a governance risk.

Word-by-word speaking lines:
“Good morning ma’am. Our project addresses a practical gap in model monitoring.  
Today, most teams track performance metrics like AUC.  
But in high-stakes domains like credit scoring, it is also critical that model reasoning remains stable over time.  
Our central question is: can explanation drift appear earlier than performance drift?  
If yes, then we get an early warning signal before model failure becomes visible in AUC.”

**Slide 2: Why This Idea Is Good**
On slide:
1. Business need: avoid silent risk in production.
2. Regulatory need: explainability consistency matters.
3. Technical value: adds a second monitoring axis beyond accuracy.

Word-by-word speaking lines:
“This idea is useful for three reasons.  
First, business safety: a model can keep decent AUC but start relying on different features in ways we do not expect.  
Second, compliance: in credit decisions, explanation consistency is important for audit and trust.  
Third, technical contribution: we are not replacing AUC, we are improving monitoring by adding explanation stability as a second axis.  
So this project solves a real deployment problem, not just an academic curiosity.”

**Slide 3: Method in One Pipeline**
On slide:
1. Train on earliest time window.
2. Freeze model.
3. Evaluate future windows.
4. Compute AUC/F1 and TESI each window.
5. Compare trends.

Word-by-word speaking lines:
“Our pipeline is simple and realistic.  
We train the model on the earliest time window, then freeze it to simulate deployment.  
Next, we evaluate future windows sequentially.  
For each window, we compute predictive metrics like AUC and explanation stability using TESI.  
Then we compare how AUC changes versus how TESI changes over time.  
This setup directly tests whether explanation drift appears earlier.”

**Slide 4: Core Metric and Claim**
On slide:
1. TESI combines attribution direction similarity and rank consistency.
2. High TESI means stable explanations.
3. Falling TESI with stable AUC means early warning.

Word-by-word speaking lines:
“Our main metric is TESI, the Temporal Explanation Stability Index.  
It combines two signals: how aligned the attribution vectors are, and whether feature-importance ranking is preserved.  
If TESI stays high, explanations are stable.  
If TESI drops while AUC is still acceptable, it means the model’s internal reasoning is shifting before performance visibly degrades.  
That is our key claim: explanation drift can be an earlier warning than performance drift.”

**Slide 5: Evidence and Credibility**
On slide:
1. Observed trend: TESI declines earlier/faster than AUC.
2. Checked across datasets/windows.
3. Can plug in accepted models like XGBoost/LightGBM.

Word-by-word speaking lines:
“Our evidence shows the same pattern repeatedly: TESI declines earlier or faster than AUC across later windows.  
That means we can detect hidden instability sooner.  
Also, this framework is model-agnostic.  
Even if we swap in widely accepted industry models such as XGBoost or LightGBM, the monitoring logic remains the same.  
So the strength of this project is not tied to one specific model architecture.”

**Slide 6: Practical Impact and Final Convincer**
On slide:
1. Proposed production rule: if TESI drops below threshold, trigger audit/retrain.
2. Contribution: proactive model governance.
3. Message: better trust, safer deployment.

Word-by-word speaking lines:
“Our final contribution is operational.  
We propose a simple governance rule: if TESI falls below a threshold, trigger model audit or retraining, even if AUC has not yet crashed.  
So this project gives proactive monitoring, not reactive firefighting.  
In short, we improve trust and safety of credit ML systems by monitoring how the model reasons, not only how it scores.”

**Possible Questions and Strong Answers**
1. “Why not just monitor AUC?”
“Because AUC is an outcome metric and can lag. TESI captures shifts in reasoning, so it can warn earlier.”

2. “Is this only for your chosen model?”
“No. The framework is model-agnostic. We can apply the same temporal monitoring to XGBoost, LightGBM, neural nets, or others.”

3. “How do we know this is not random fluctuation?”
“We track trends across multiple windows and compare to baseline. Consistent directional decline in TESI gives stronger evidence than one-off noise.”

4. “What is the practical use in industry?”
“It becomes an MLOps alert. If TESI drops below threshold, teams investigate feature drift, fairness risk, and retraining need.”

5. “Are you claiming TESI replaces AUC?”
“No. It complements AUC. AUC tells performance quality; TESI tells explanation stability. Both are needed.”

6. “Why is this important in finance specifically?”
“Credit decisions are regulated and high-impact. Stable, auditable reasoning is essential for trust and compliance.”

7. “Could explanation method choice bias results?”
“We can validate with multiple explainers. If trend persists across methods, confidence increases.”

8. “What is your single biggest contribution?”
“A practical early-warning framework for model staleness using explanation drift, before obvious performance degradation.”

**30-Second Emergency Version**
“Our project adds a missing layer to model monitoring. In credit risk, AUC alone is not enough because model reasoning can drift before accuracy drops. We measure that with TESI across time windows using a frozen deployment-style setup. We consistently observe TESI degrading earlier than AUC, so it acts as an early warning for audit or retraining. This is practical, model-agnostic, and directly useful for safer financial AI governance.”

If you want, I can now give you a “mock viva drill” with 15 rapid-fire tough questions and one-line winning replies to memorize tonight.