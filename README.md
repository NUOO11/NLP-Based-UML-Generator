# NLP-Driven Automated UML Modeling System
**(Final  Project)**

> **Author:** [Xing Nuo]
> **Student ID:** [GS70842]
> **Date:** February 2026
> **Repository:** [Link to your GitHub if available]

---

##  1. Project Overview

This software artifact implements an automated pipeline that extracts structural and behavioral information from natural language requirement .

It utilizes **Natural Language Processing (NLP)** (via the `spaCy` library) to parse text and utilizes the **Kroki.io API** to render UML diagrams without requiring local Graphviz installations.

**Key Features:**
* **Input:** Supports `.txt`, `.pdf` (with OCR auto-repair), and `.docx`.
* **Output:** Generates Class, Sequence, Use Case, and Activity Diagrams.
* **Validation:** Includes a dashboard to compare generated models against a ground truth.

---

##  2. Prerequisites (Environment Setup)

Before running the code, ensure you have the following installed:

1.  **Python 3.8 or higher**:
    * Download from: [python.org](https://www.python.org/downloads/)
    * *Important:* During installation on Windows, check the box **"Add Python to PATH"**.
2.  **Internet Connection**:
    * Required for downloading the NLP model and for the cloud-based diagram rendering service.

---

## 🚀 3. Installation Guide (Step-by-Step)

Follow these steps to set up the project from scratch.

### Step 1: Prepare the Project Folder
1.  Unzip the project file (or clone the repository).
2.  Open your **Terminal** (macOS/Linux) or **Command Prompt / PowerShell** (Windows).
3.  Navigate to the project folder:
    ```bash
    cd path/to/your/project_folder
    ```

### Step 2: Create a Virtual Environment (Recommended)
Using a virtual environment prevents conflicts with other system libraries.

* **For Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *(You will see `(venv)` appear at the start of your command line)*

* **For macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### Step 3: Install Python Dependencies
Run the following command to install all required libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt


### Step 4: Download the NLP Model (Critical)
The system requires a specific English language model from spaCy to analyze grammar. You must run this command:

Bash
python -m spacy download en_core_web_lg
(This downloads the "Large" English model, approx. 500MB+)

4. How to Run the Application
Once the installation is complete, start the application by running:

Bash
streamlit run thesis_uml_architect.py
(Note: Replace thesis_uml_architect.py with your actual Python filename if it is different)

What happens next?

The terminal will show a "Local URL" (usually http://localhost:8501).

Your default web browser should open automatically loading the system interface.

🛠️ 5. User Manual
1. Upload Requirements
In the Sidebar (Left Panel), locate the "Upload  Document" button.

Supported formats: PDF, DOCX, or TXT.

Note: If you upload a PDF, the system automatically applies a repair algorithm to fix common OCR issues (like missing spaces).

2. Select Diagram Type
Choose one of the following from the dropdown menu:

Class Diagram: For structural entities and relationships.

Sequence Diagram: For actor-system interactions.

Use Case Diagram: For user stories ("As a User, I want to...").

Activity Diagram: For logical flows (If/Else conditions).

3. Generate & View
Click the "🚀 Generate UML Model" button.

Tab 1 (Visual Model): Displays the rendered diagram (PNG). You can download it here.

Tab 2 (PlantUML Source): Displays the generated code.

4. Evaluation (Thesis Validation)
Scroll to the bottom "Thesis Validation Dashboard".

Paste the "Ground Truth" (Correct) PlantUML code into the text area.


The system will calculate Precision, Recall, and F1-Score for entities and relations.
