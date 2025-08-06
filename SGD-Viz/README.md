# SGD vs GD Loss Landscape Visualizer

This interactive Streamlit app visualizes how **Stochastic Gradient Descent (SGD)** navigates the noisy batch-wise loss landscape compared to the global loss surface. It helps understand the optimization process step-by-step by combining **3D**, **2D contour**, and **loss convergence** plots.

---

## 📌 Features

- ✅ Visualizes **global vs. batch loss surfaces** in 3D
- ✅ Shows the **SGD path** taken across batches
- ✅ Batch-level vs. global loss **contour plot** (2D)
- ✅ Tracks and plots **loss convergence**
- ✅ Adjustable **learning rate** and **batch size**
- ✅ One-click **epoch training** and **model reset**
- ✅ Metrics on **current step**, **epoch**, **loss gap**, and more

---

## 🚀 Getting Started

### 1. Clone this repo (if applicable)

```bash
git clone https://github.com/your-username/sgd-visualizer.git
cd sgd-visualizer
````

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## Learning Outcome

This app is ideal for:

* Data science students learning about **SGD**
* ML researchers exploring **batch effects**
* Educators demonstrating **loss landscapes**
* Anyone curious about how noisy gradients still lead to convergence!

---

## Acknowledgements
Built with ❤️ using [Streamlit](https://streamlit.io/)
