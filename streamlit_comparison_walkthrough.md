# Streamlit Multi-Model Comparison Dashboard

We have updated the Streamlit frontend in [app.py](file:///e:/15-06-26/Fine-tune%20BERT%20(deepsetgbert-large)/app.py) to represent all 7 trained GBERT-large model configurations. 

This enables you to:
1. **Choose an Active Model Category**: Use the sidebar selectbox to inspect details of any model configuration and test it.
2. **Perform Side-by-Side Model Comparison**: Click **Compare All Models 🔍** to run predictions across all 7 models simultaneously, displaying a comparative results table.

---

## 📸 Interface Preview & Demo

Here is a screenshot of the multi-model comparison interface showing predictions side-by-side:

![Streamlit Comparison Table](file:///C:/Users/vijayakr/.gemini/antigravity-ide/brain/794c216e-2fc1-4399-b472-4f13a9108772/comparison_table_1783554906274.png)

### Video Walkthrough of Verification Session
Below is the full animation recording of our browser verification session showing the model selector and comparison run:

![Multi-Model Verification Session](file:///C:/Users/vijayakr/.gemini/antigravity-ide/brain/794c216e-2fc1-4399-b472-4f13a9108772/streamlit_multi_model_test_1783554522619.webp)

---

## 🛠️ Supported Model Categories

The selector supports the following trained models:
1. **v5 GBERT-large (Clean / Robust)**: `models/v5_best_model_clean` (Length-stratified & template-stripped)
2. **v5 GBERT-large (Baseline)**: `models/v5_best_model` (Baseline with templates)
3. **v1 GBERT-large (Best Model)**: `models/best_model` (First successful training run)
4. **v1 GBERT-large (Full Model)**: `models/full_model` (Trained on the full 57k run)
5. **500k GBERT-large (Collapsed)**: `models/full_model_500k` (Model that experienced collapse)
6. **Legal GBERT-large (Law Focus)**: `models/legal_model` (Fine-tuned on legal text style)
7. **100k GBERT-large (Intermediate)**: `models/model_100k` (Trained on the 100k dataset split)

---

## 🚀 How to Run the App

If you want to view the app in an external browser, run the following command in your terminal and navigate to the Local URL:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```
*Local URL: [http://localhost:8501](http://localhost:8501)*
