# LinkedIn post draft (personalize the links, then post Tuesday–Thursday morning)

---

I applied to a lot of internships where my resume said "machine learning" but I had nothing to *show* for it. So I built something to show.

📉 Customer Churn Predictor — an end-to-end ML project that predicts which telecom customers will leave, and explains WHY for each individual customer using SHAP.

What it does:
▶️ Type in a customer's contract, charges, tenure…
▶️ Get a churn probability + a ranked chart of what's driving it (e.g., "month-to-month contract pushes risk up; 2-year tenure pulls it down")

How I built it:
🔹 Python, pandas, scikit-learn, XGBoost
🔹 Compared 3 models by cross-validated AUC (class imbalance ruled out accuracy)
🔹 Explained every prediction with SHAP — because a prediction without a reason doesn't help a retention team
🔹 Deployed as an interactive app so anyone can try it

The part nobody warns you about: the dataset had a column of numbers secretly stored as strings. Cleaning that taught me more than the modeling.

🔗 Try the live demo: [your Hugging Face Space link]
🔗 Code on GitHub: [your GitHub repo link]

Feedback welcome — especially if you work in data science and would have done something differently. #DataScience #MachineLearning #OpenToWork

---

**Posting checklist**
- [ ] Train the model first, fill the real AUC numbers into the post
- [ ] Deploy to Hugging Face Spaces, paste the live link
- [ ] Post Tue–Thu morning; reply to every comment
- [ ] Add both links to your LinkedIn Featured section
