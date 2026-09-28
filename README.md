# 🚀 PC3 Software Defect Prediction Dashboard

Every software engineering team knows the dread of shipping code only to discover critical bugs in production. This project was born from a desire to tackle that exact challenge head-on: using data science to look into the past, evaluate the present, and predict software defects before they ever make it to a live release.

## 📖 The Story Behind the Data

To build this solution, I turned to the well-known **NASA PROMISE PC3 dataset**, a benchmark repository rich with code complexity metrics. Raw code metrics can often feel like an overwhelming sea of numbers—lines of code, operator counts, and branch densities. My goal was to transform these dry metrics into a living, breathing story about software quality.

I set out to answer a core question: *Can we look at historical code patterns to accurately forecast which modules are destined to fail?*

## 🔍 Journey Through the Dashboard

To make the analysis intuitive and actionable, I structured the dashboard around a temporal narrative—moving from historical hindsight to real-time foresight:

* **⏳ The Past (Exploratory Data Analysis):** 
  Before building models, I had to understand the historical landscape. Diving into the EDA tab revealed deep insights into our 1,563 code modules. I discovered that defect rates hover around 10.2%, and uncovered the top metrics most heavily correlated with code failures. It became clear that certain complexity indicators are silent red flags for underlying bugs.

* **📊 The Present (Model Performance):** 
  With the historical patterns understood, I trained and evaluated machine learning classification models to see how well they identify defect-prone modules in real time. This section breaks down the reliability of the models, tracking how accurately algorithms can distinguish between clean code and high-risk modules.

* **🔮 The Future (Module Risk Assessor):**
  The most exciting part of this project is the interactivity. Instead of just looking at static charts of past data, engineering teams can use the live risk simulator. By inputting custom code metrics into the tool, developers can instantly test a new module and predict its defect risk *before* deployment, shifting engineering from reactive bug-fixing to proactive quality assurance.

## 🛠️ Built With
* **Python** & **Pandas** for data wrangling and processing.
* **Scikit-Learn** for machine learning classification.
* **Streamlit** for bringing the interactive, multi-tab interface to life.
* **Matplotlib** & **Seaborn** for clear, narrative-driven visualizations.

Crafted with code, creativity and love ❤️ you can check out the live dashboard app in the PC3 Streamlit link🔗 above, thank you 🙏 
proudly 🦚 Chioma's Art 🎨.
