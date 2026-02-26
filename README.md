# 📊 NLP Graphic Tool

A modern NLP processing tool for **Tigrinya (ትግርኛ)** and **English** text. 

**🚀 Live Demo:** [https://weldegebriel-tesfayhagos-nlp.streamlit.app/](https://weldegebriel-tesfayhagos-nlp.streamlit.app/)

---

## 📖 How to Use

1. **Access the App:** Open the [Live Demo](https://weldegebriel-tesfayhagos-nlp.streamlit.app/) or run locally.
2. **Input Text:** 
   - Paste your text directly into the **Text Area**.
   - Or **Upload** a `.txt` or `.md` file.
   - You can also load built-in **Tigrinya/English samples** from the sidebar.
3. **Configure Pipeline:** (Optional) Use the sidebar to toggle:
   - Language Auto-detection
   - Stopword Elimination (Requirement 2.1)
   - Lemmatization (Requirement 2.2)
4. **Analyze:** Click **🚀 Analyze Document** to process the text.
5. **Explore Results:** Navigate through the interactive tabs:
   - **Frequencies:** View term counts (Requirement 2.3).
   - **Distances:** Analyze term positions relative to document boundaries (Requirement 2.4).
   - **Compound Relevance:** See weighted importance scores (Requirement 2.5).
   - **Visualizations:** Interactive charts and pie plots.
6. **Export:** Download your results as **CSV** or **JSON**.

---

## 🖼️ User Interface

### Analysis Overview
![Dashboard Overview](doc/screenshot_overview.png)

### Interactive Insights
![Interactive Charts](doc/screenshot_insights.png)

---

## 🛠️ Local Setup

If you wish to run the tool locally:

```bash
# Clone the repository
git clone <your-repo-url>
cd nlp_graphic_tool

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📝 Features & Requirements

- ✅ **Stopword Removal (2.1):** Support for English and Tigrinya.
- ✅ **Lemmatization (2.2):** Standard English lemmatizer.
- ✅ **Frequencies (2.3):** Full term count calculations.
- ✅ **Distance Metrics (2.4):** Early/Late position analysis.
- ✅ **Compound Relevance (2.5):** 50% Frequency + 50% Earliness.
- 🌍 **Dual Language:** Smart detection of Tigrinya and English.
- 📊 **Visualizations:** Powered by Plotly.

---
*Developed for NLP Research Assignment Option 2.*
