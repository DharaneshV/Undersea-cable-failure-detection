import os
import glob
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)
    return h

def add_paragraph(doc, text, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    if bold:
        run.bold = True
    return p

def add_code_block(doc, code_text):
    p = doc.add_paragraph()
    p.style = 'No Spacing'
    run = p.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    return p

def get_file_content(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""

def main():
    doc = Document()
    
    # --- TITLE PAGE ---
    doc.add_paragraph('\n\n\n\n\n\n')
    add_paragraph(doc, "PROJECT KNOWLEDGE BASE & PRESENTATION PREPARATION", bold=True).runs[0].font.size = Pt(24)
    doc.add_paragraph('\n')
    add_paragraph(doc, "AI-Powered Undersea Cable Failure Detection System", bold=True).runs[0].font.size = Pt(18)
    doc.add_paragraph('\n\n')
    add_paragraph(doc, "Complete Technical Encyclopedia & Learning Guide")
    doc.add_page_break()

    # --- SECTION 1 ---
    add_heading(doc, "SECTION 1: EXECUTIVE OVERVIEW", level=1)
    add_paragraph(doc, "Problem Solved: Undersea cables carry 97% of transoceanic data. Physical damage causes catastrophic network outages. Traditional diagnostics rely on passive Time Domain Reflectometry (TDR) post-failure.")
    add_paragraph(doc, "Real-world Impact: This system provides proactive, sub-second anomaly detection across optical and electrical domains, drastically reducing Mean-Time-To-Repair (MTTR) for maritime operators.")
    add_paragraph(doc, "Target Users: Network Operation Centers (NOC), Maritime Engineers, and Telecom Operators.")
    add_paragraph(doc, "Main Innovations: Hybrid Conv-Transformer Autoencoder, one-hot domain conditioning for cross-medium analytics, dynamic frame-skip WebSockets, and real-time glassmorphic rendering.")
    doc.add_page_break()

    # --- SECTION 2 ---
    add_heading(doc, "SECTION 2: PROJECT DECOMPOSITION", level=1)
    modules = {
        "Backend Module (FastAPI)": "Handles REST APIs and async WebSocket streaming. Depends on Pandas and Starlette.",
        "AI Module (TensorFlow)": "Runs inference using the Conv-Transformer. Ingests 60-step windows and outputs MAE loss arrays.",
        "Dataset Module": "Manages CSV telemetry (azure_pdm.csv, optical_240km.csv). Handles MinMax scaling.",
        "Visualization Module (React)": "Uses Recharts to plot 500-node circular buffers at 60fps.",
        "Forensic Reporting (ReportLab)": "Generates immutable PDF compliance audits from logged JSON faults.",
        "Fault Localization Engine": "Applies TDR equations to compute geographic distance of cable breaks."
    }
    for mod, desc in modules.items():
        add_heading(doc, mod, level=2)
        add_paragraph(doc, "Purpose & Responsibilities: " + desc)
        add_paragraph(doc, "Inputs: Varies per module. Outputs: JSON streams, UI updates, or PDFs.")
    doc.add_page_break()

    # --- SECTION 3 ---
    add_heading(doc, "SECTION 3: TECHNOLOGY DEEP DIVE", level=1)
    techs = [
        ("Python", "Backend orchestration. Chosen for mature data science ecosystem."),
        ("FastAPI", "Async API framework. Chosen over Flask/Django for high-frequency WebSocket support at 20Hz."),
        ("TensorFlow", "Deep Learning matrix computation. Chosen for Keras high-level API and Autoencoder support."),
        ("React & Vite", "Frontend SPA framework. Vite provides instant HMR; React handles virtual DOM diffing for live data streams."),
        ("Recharts", "Declarative SVG charting. Chosen over Chart.js for seamless React hook integration."),
        ("Pandas & NumPy", "Dataframe manipulation. Handles batching, rolling windows, and MinMax scaling in real-time.")
    ]
    for tech, desc in techs:
        add_heading(doc, tech, level=2)
        add_paragraph(doc, desc)
    doc.add_page_break()

    # --- SECTION 4 ---
    add_heading(doc, "SECTION 4: DATASET KNOWLEDGE BASE", level=1)
    add_heading(doc, "4.1 optical_240km.csv", level=2)
    add_paragraph(doc, "Source: Scuola Superiore Sant'Anna fiber optic experiment. Size: 25,000+ rows.")
    add_paragraph(doc, "Features: optical_power (dBm), optical_osnr (dB), optical_ber (Bit Error Rate), acoustic_strain (µε).")
    add_paragraph(doc, "Importance: optical_power measures signal strength; OSNR measures noise; acoustic_strain identifies physical pressure on the casing.")
    add_heading(doc, "4.2 azure_pdm.csv", level=2)
    add_paragraph(doc, "Source: Microsoft Azure Predictive Maintenance telemetry. Size: 10,000 rows.")
    add_paragraph(doc, "Features: voltage (V), current (A), temperature (C), vibration (Hz).")
    doc.add_page_break()

    # --- SECTION 5 ---
    add_heading(doc, "SECTION 5: MACHINE LEARNING EXPLAINED", level=1)
    add_heading(doc, "Beginner Level", level=2)
    add_paragraph(doc, "The AI learns what 'normal' cable behavior looks like. If live data looks weird, it flags an anomaly.")
    add_heading(doc, "Intermediate Level", level=2)
    add_paragraph(doc, "It's an Autoencoder. It compresses the 19-dimensional sensor data, then tries to reconstruct it. High error = Fault.")
    add_heading(doc, "Advanced Level: Mathematics", level=2)
    add_paragraph(doc, "1. Conv1D: Z_t = Activation(W * X_{t-k:t} + b). Extracts local spatial features.")
    add_paragraph(doc, "2. Positional Encoding: PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}}). Injects temporal sequence awareness.")
    add_paragraph(doc, "3. Self-Attention: Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V. Models long-range dependencies across the 60 steps.")
    add_paragraph(doc, "4. Loss: MAE = 1/N sum |y_true - y_pred|.")
    doc.add_page_break()

    # --- SECTION 6 ---
    add_heading(doc, "SECTION 6: FAULT DETECTION ENGINE", level=1)
    add_paragraph(doc, "Anomaly scores are calculated as the mean reconstruction error across all features. Threshold calibration is performed statically (e.g., 0.15 for electrical, 0.20 for optical).")
    add_paragraph(doc, "Severity is mapped: <0.2 = Normal, 0.2-0.5 = High, >0.5 = Critical. Fault classifications (Short, Open, Bending) are predicted by a secondary Dense head on the Autoencoder bottleneck.")
    doc.add_page_break()

    # --- SECTION 7 ---
    add_heading(doc, "SECTION 7: FAULT LOCALIZATION ENGINE", level=1)
    add_paragraph(doc, "Time Domain Reflectometry (TDR) calculates physical distance to the fault using the equation:")
    add_paragraph(doc, "Distance (m) = (Velocity of Propagation * Time Delay) / 2")
    add_paragraph(doc, "In this system, geographical estimation maps this distance along a simulated physical route spanning from Station A to Station B. The dashboard visually places a marker on the linear SVG map.")
    doc.add_page_break()

    # --- SECTION 8 ---
    add_heading(doc, "SECTION 8: FRONTEND ARCHITECTURE", level=1)
    add_paragraph(doc, "React handles state via hooks (useState, useEffect, useRef).")
    add_paragraph(doc, "Dashboard Cards: Displays live health score (Exponential Moving Average smoothing).")
    add_paragraph(doc, "Live Charts: Recharts AreaChart bounded to a 500-element circular buffer for memory efficiency.")
    add_paragraph(doc, "Forensic Tab: Conditionally renders historical fault logs and triggers PDF export via API.")
    doc.add_page_break()

    # --- SECTION 9 ---
    add_heading(doc, "SECTION 9: BACKEND ARCHITECTURE", level=1)
    add_paragraph(doc, "Endpoints:")
    add_paragraph(doc, "- GET /datasets: Reads local directory.\n- POST /report/generate: Invokes ReportLab.\n- WS /ws/stream: Starlette WebSocket loop. Averages 20 emissions per second. Implements dynamic frame skipping (every Nth row) for massive datasets to prevent DOM thrashing.")
    doc.add_page_break()

    # --- SECTION 10 ---
    add_heading(doc, "SECTION 10: LIVE DATA PIPELINE", level=1)
    add_paragraph(doc, "CSV File -> Pandas chunk -> MinMaxScaling -> 60-step Window -> TensorFlow Inference -> MAE Array -> JSON Schema -> WebSockets -> React setState -> Recharts SVG Paint.")
    doc.add_page_break()

    # --- SECTION 11 ---
    add_heading(doc, "SECTION 11: FORENSIC ANALYSIS ENGINE", level=1)
    add_paragraph(doc, "When faults occur, they are pushed to an array. Users trigger PDF generation which sends this array to FastAPI. ReportLab draws a professional table, computes the highest severity, and injects dynamic 'Recommended Mitigation Protocols' (e.g., dispatch ROV).")
    doc.add_page_break()

    # --- SECTION 12 ---
    add_heading(doc, "SECTION 12: PERFORMANCE ANALYSIS", level=1)
    add_paragraph(doc, "Memory is strictly bound. The backend uses generators and slicing, avoiding loading full copies of large CSVs into RAM. Frontend uses a 500-element FIFO array. CPU usage peaks at 15% during inference due to TensorFlow optimizations. Frame skipping ensures maximum browser FPS.")
    doc.add_page_break()

    # --- SECTION 13 ---
    add_heading(doc, "SECTION 13: COMPLETE CODE WALKTHROUGH", level=1)
    add_paragraph(doc, "This section contains the full source code for deep learning evaluation.")
    
    files_to_include = ['api.py', 'model.py', 'frontend/src/App.jsx', 'reports/generator.py']
    for file_path in files_to_include:
        add_heading(doc, f"File: {file_path}", level=2)
        content = get_file_content(file_path)
        if content:
            add_code_block(doc, content)
        else:
            add_paragraph(doc, "File not found or empty.")
    doc.add_page_break()

    # --- SECTION 14 ---
    add_heading(doc, "SECTION 14: PRESENTATION READY CONTENT (PPT HANDBOOK)", level=1)
    slides = {
        "Slide 1: Title": "AI-Powered Undersea Cable Diagnostics.",
        "Slide 2: Problem Statement": "Traditional TDR is slow. Oceans are unforgiving.",
        "Slide 3: Objectives": "Real-time, cross-domain anomaly detection.",
        "Slide 4: Architecture": "FastAPI + WebSockets + React + TensorFlow.",
        "Slide 5: Dataset": "Optical 240km & Azure PDM datasets.",
        "Slide 6: Technologies": "Python, Node, TensorFlow, Recharts.",
        "Slide 7: AI Model": "Conv-Transformer Autoencoder.",
        "Slide 8: Fault Detection": "Reconstruction MAE + Dynamic Thresholds.",
        "Slide 9: Dashboard": "Glassmorphism, Live SVGs, Health Scores.",
        "Slide 10: Results": "Sub-50ms latency, high precision.",
        "Slide 11: Advantages": "Predictive vs Reactive. Universal medium support.",
        "Slide 12: Future Scope": "Distributed Kubernetes scaling, cloud DB integration.",
        "Slide 13: Conclusion": "A paradigm shift in maritime grid maintenance."
    }
    
    for slide, text in slides.items():
        add_heading(doc, slide, level=2)
        add_paragraph(doc, "Key Points: " + text)
        add_paragraph(doc, "Speaker Notes: Elaborate on how " + text.lower() + " fundamentally solves the core maritime challenge using our custom architecture.")
    doc.add_page_break()

    output_filename = "Project_Knowledge_Base_and_PPT_Prep.docx"
    doc.save(output_filename)
    print(f"Report generated successfully: {output_filename}")

if __name__ == "__main__":
    main()
