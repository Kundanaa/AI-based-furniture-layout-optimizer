# AI-based furniture layout optimizer


#### **Project Overview**

This project is an AI-based furniture layout optimizer that generates a structured 2D layout for a room based on predefined constraints such as furniture dimensions, spacing rules, and functional zones. The application ensures no overlaps between furniture items and places them realistically within the boundaries of the room.

---

#### **Prerequisites**

1. **Python Version**: Ensure Python 3.8 or higher is installed.
2. **Required Libraries**:
    - Install the following libraries using `pip`:

```bash
pip install numpy torch matplotlib streamlit
```


---


#### **Steps to Run the Application**

1. **Clone the Repository**:

```bash
git clone https://github.com/Kundanaa/Insydeio_AIMLIntern_TechAssignment.git

```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

3. **Train the Model**:
Run the training script to generate the optimized model:

```bash
python train.py
```

    - This will save the trained model as `layout_model.pth`.
4. **Run the Application**:
Launch the Streamlit application to visualize layouts:

```bash
streamlit run app.py
```

    - Open the URL provided by Streamlit in your browser (e.g., `http://localhost:8501`).
5. **Interact with the Application**:
    - Specify room dimensions, number of furniture items, and other parameters.
    - Generate optimized layouts and visualize them.

---
