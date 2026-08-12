
# -*- coding: utf-8 -*-
# =============================================================================
# SignalSense AI – Intelligent Signal Quality Analyzer
# Powered by IBM watsonx.ai Granite Models
# Multi-Agent Agentic AI Architecture
# =============================================================================

import os
import io
import csv
import json
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Load environment variables from .env file (if python-dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — environment variables must be set manually

app = Flask(__name__)

# =============================================================================
# IBM watsonx.ai Configuration
# Credentials are loaded from .env file or system environment variables.
# Set WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL in your .env file.
# =============================================================================
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")

def get_model():
    """Initialize and return the IBM watsonx.ai Granite model instance."""
    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 0.9,
        }
    )
    return model

def generate_response(prompt: str) -> str:
    """
    Core function that calls IBM watsonx.ai Granite Model.
    All four agents use this function to generate AI responses.
    """
    try:
        model = get_model()
        # ---- IBM watsonx.ai API call ----
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else "No response generated."
    except Exception as e:
        return f"[watsonx.ai Error] {str(e)}\n\nPlease check your WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL environment variables."

# =============================================================================
# AGENT 1 – Signal Data Analysis Agent
# Analyzes signal performance metrics and generates quality summaries
# =============================================================================
def signal_analysis_agent(data: dict) -> str:
    """
    Agent 1: Analyze signal quality parameters using IBM Granite Model.
    Generates signal quality summary, parameter analysis, and performance trends.
    """
    prompt = f"""You are a senior telecommunications engineer and signal quality analyst.
Analyze the signal quality parameters below and write a professional engineering report.

IMPORTANT INSTRUCTIONS:
- Do NOT repeat or echo back the raw input data or CSV rows in your response.
- Do NOT use LaTeX, math boxes, or $\\boxed{{}}$ notation.
- Write in plain professional English with clear headings and bullet points.
- Be concise, technical, and actionable.

Signal Parameters:
- SNR: {data.get('snr', 'N/A')} dB
- BER: {data.get('ber', 'N/A')}
- Latency: {data.get('latency', 'N/A')} ms
- Throughput: {data.get('throughput', 'N/A')} Mbps
- Packet Loss: {data.get('packet_loss', 'N/A')} %
- RSSI: {data.get('rssi', 'N/A')} dBm
- Jitter: {data.get('jitter', 'N/A')} ms
- Frequency Band: {data.get('frequency_band', 'N/A')}
- Notes / Environment: {data.get('notes', 'None')}
- Historical Trend Summary: {data.get('historical', 'Not provided')}

Report Structure (use these exact headings):

## 1. Overall Signal Quality Assessment
State the quality rating: Excellent / Good / Fair / Poor — with a one-sentence justification.

## 2. Parameter-by-Parameter Analysis
For each parameter, state the value, whether it is within acceptable range, and its impact.

## 3. Key Performance Observations
List the 3–5 most critical observations affecting communication quality.

## 4. Communication Health Summary
A short paragraph summarising the overall health of the communication link.

## 5. Trends & Patterns
Describe any degradation trends or anomalies identified from the historical data."""
    # ---- Calling IBM watsonx.ai Granite Model for Signal Analysis ----
    return generate_response(prompt)

# =============================================================================
# AGENT 2 – Signal Issue Detection Agent
# Detects communication issues, interference, and signal degradation
# =============================================================================
def signal_issue_detection_agent(data: dict) -> str:
    """
    Agent 2: Detect communication issues and anomalies using IBM Granite Model.
    Identifies root causes, severity, and communication risk indicators.
    """
    prompt = f"""You are an expert network diagnostics engineer specializing in signal anomaly detection.
Analyze the signal data below and produce a structured issue detection report.

IMPORTANT INSTRUCTIONS:
- Do NOT repeat or echo back the raw input data or CSV rows in your response.
- Do NOT use LaTeX, math boxes, or $\\boxed{{}}$ notation.
- Write in plain professional English with clear headings and bullet points.
- Be direct and technically precise.

Signal Parameters:
- SNR: {data.get('snr', 'N/A')} dB
- BER: {data.get('ber', 'N/A')}
- Latency: {data.get('latency', 'N/A')} ms
- Throughput: {data.get('throughput', 'N/A')} Mbps
- Packet Loss: {data.get('packet_loss', 'N/A')} %
- RSSI: {data.get('rssi', 'N/A')} dBm
- Jitter: {data.get('jitter', 'N/A')} ms
- Frequency Band: {data.get('frequency_band', 'N/A')}
- Reported Symptoms: {data.get('symptoms', 'None reported')}
- Environment: {data.get('environment', 'Not specified')}

Report Structure (use these exact headings):

## 1. Detected Signal Issues
List each identified issue with a brief description.

## 2. Root Cause Analysis
For each issue, explain the most likely technical root cause.

## 3. Severity Assessment
Rate each issue: Critical / High / Medium / Low — with justification.

## 4. Communication Risk Indicators
Summarise the key risks to communication reliability.

## 5. Interference & Anomaly Patterns
Describe any interference sources or abnormal behaviour patterns detected."""
    # ---- Calling IBM watsonx.ai Granite Model for Issue Detection ----
    return generate_response(prompt)

# =============================================================================
# AGENT 3 – Intelligent Insight & Recommendation Agent
# Generates AI-powered engineering recommendations to improve signal quality
# =============================================================================
def signal_insight_agent(data: dict) -> str:
    """
    Agent 3: Generate intelligent recommendations using IBM Granite Model.
    Provides signal optimization strategies, noise reduction, antenna optimization.
    """
    prompt = f"""You are an expert telecommunications systems engineer and signal optimization specialist.
Based on the signal data and network context below, generate a concise engineering recommendations report.

IMPORTANT INSTRUCTIONS:
- Do NOT repeat or echo back the raw input data in your response.
- Do NOT use LaTeX, math boxes, or $\\boxed{{}}$ notation.
- Write in plain professional English with clear headings and bullet points.
- Prioritise recommendations by impact (highest impact first).

Current Signal Metrics:
- SNR: {data.get('snr', 'N/A')} dB | BER: {data.get('ber', 'N/A')} | Latency: {data.get('latency', 'N/A')} ms
- Throughput: {data.get('throughput', 'N/A')} Mbps | Packet Loss: {data.get('packet_loss', 'N/A')} %
- RSSI: {data.get('rssi', 'N/A')} dBm | Jitter: {data.get('jitter', 'N/A')} ms

Network Context:
- Type: {data.get('network_type', 'Not specified')} | Band: {data.get('frequency_band', 'N/A')}
- Environment: {data.get('environment', 'Not specified')}
- Application Requirements: {data.get('app_requirements', 'General use')}
- Known Problems: {data.get('problems', 'None specified')}

Report Structure (use these exact headings):

## 1. Signal Optimization Strategies
Highest-impact actions to immediately improve signal quality.

## 2. Noise Reduction & Interference Mitigation
Specific techniques to reduce noise floor and EMI.

## 3. Antenna & RF Optimization
Antenna alignment, gain, polarisation, and placement recommendations.

## 4. Frequency & Channel Planning
Band selection, channel allocation, and spectrum management advice.

## 5. Network Configuration Improvements
Protocol, QoS, power, and configuration parameter tuning.

## 6. Equipment & Infrastructure Upgrades
Hardware upgrade recommendations if current equipment is a limiting factor.

## 7. Implementation Priority Summary
A ranked action list: what to do first, second, and third."""
    # ---- Calling IBM watsonx.ai Granite Model for Recommendations ----
    return generate_response(prompt)

# =============================================================================
# AGENT 4 – Predictive Signal Failure Agent
# Forecasts future signal degradation and recommends preventive actions
# =============================================================================
def predictive_failure_agent(data: dict) -> str:
    """
    Agent 4: Predict future signal failures using IBM Granite Model.
    Generates failure probability, risk assessment, and preventive maintenance plans.
    """
    prompt = f"""You are a predictive analytics specialist in telecommunications and network reliability engineering.
Analyse the signal trend data below and produce a professional predictive failure report.

IMPORTANT INSTRUCTIONS:
- Do NOT repeat or echo back the raw input data, CSV rows, or measurement tables in your response.
- Do NOT use LaTeX, math boxes, or $\\boxed{{}}$ notation.
- Write in plain professional English with clear headings and bullet points.
- Base your predictions on the trends, not just individual data points.

Observed Trends:
- Latency Trend: {data.get('latency_trend', 'N/A')}
- Throughput Trend: {data.get('throughput_trend', 'N/A')}
- Packet Loss Trend: {data.get('packet_loss_trend', 'N/A')}
- Current SNR: {data.get('snr', 'N/A')} dB | Current BER: {data.get('ber', 'N/A')}
- Equipment Age: {data.get('equipment_age', 'Unknown')}
- Environmental Conditions: {data.get('environmental_conditions', 'Normal')}
- Recent Incidents: {data.get('recent_incidents', 'None')}
- Maintenance History: {data.get('maintenance_history', 'Not provided')}
- Historical Summary: {data.get('historical_data', 'Not provided')[:500] if data.get('historical_data') else 'Not provided'}

Report Structure (use these exact headings):

## 1. Signal Degradation Forecast
Provide a short-term (7-day), mid-term (30-day), and long-term (90-day) outlook based on observed trends.

## 2. Failure Probability Assessment
Estimate the probability of a significant communication failure within 7 / 30 / 90 days (as a percentage).

## 3. Risk Classification
Overall risk level: Critical / High / Medium / Low — with justification.

## 4. Most Vulnerable Components & Links
Identify which parts of the communication chain are most at risk.

## 5. Predicted Failure Modes
List the most likely failure scenarios in order of probability.

## 6. Preventive Maintenance Schedule
Specific actions with recommended timelines (e.g., "Within 7 days: inspect antenna connectors").

## 7. Early Warning Indicators
List the key metrics to monitor closely and their alert thresholds."""
    # ---- Calling IBM watsonx.ai Granite Model for Predictive Analytics ----
    return generate_response(prompt)

# =============================================================================
# AGENT ORCHESTRATOR
# Routes user requests to the appropriate specialized agent
# =============================================================================
def orchestrator(agent_type: str, data: dict) -> str:
    """
    Orchestrator function that routes requests to the appropriate AI agent.
    Selects agent based on the feature/task requested by the user.
    """
    routing_map = {
        "analysis":    signal_analysis_agent,
        "detection":   signal_issue_detection_agent,
        "insight":     signal_insight_agent,
        "predictive":  predictive_failure_agent,
    }
    agent_fn = routing_map.get(agent_type)
    if agent_fn:
        return agent_fn(data)
    return "Unknown agent type. Please select a valid analysis feature."

# =============================================================================
# HTML TEMPLATE – Complete Bootstrap 5 UI (inline via render_template_string)
# =============================================================================
BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SignalSense AI – Intelligent Signal Quality Analyzer</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
:root{
  --ss-bg:#0a0e1a;
  --ss-sidebar:#0d1323;
  --ss-card:#111827;
  --ss-border:#1e2d45;
  --ss-accent:#00d4ff;
  --ss-accent2:#7c5cd8;
  --ss-accent3:#00ff9d;
  --ss-warn:#ffb800;
  --ss-danger:#ff4b6e;
  --ss-text:#e2e8f0;
  --ss-muted:#64748b;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--ss-bg);color:var(--ss-text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;display:flex;}
/* Sidebar */
#sidebar{width:260px;min-height:100vh;background:var(--ss-sidebar);border-right:1px solid var(--ss-border);display:flex;flex-direction:column;position:fixed;top:0;left:0;z-index:100;transition:transform .3s;}
#sidebar .brand{padding:20px 24px;border-bottom:1px solid var(--ss-border);}
#sidebar .brand h5{color:var(--ss-accent);font-weight:700;font-size:1.1rem;letter-spacing:.5px;}
#sidebar .brand small{color:var(--ss-muted);font-size:.72rem;}
#sidebar nav{flex:1;padding:16px 0;}
#sidebar nav a{display:flex;align-items:center;gap:12px;padding:11px 24px;color:var(--ss-muted);text-decoration:none;font-size:.88rem;transition:all .2s;border-left:3px solid transparent;}
#sidebar nav a:hover,#sidebar nav a.active{color:var(--ss-accent);background:rgba(0,212,255,.07);border-left-color:var(--ss-accent);}
#sidebar nav a i{font-size:1rem;width:20px;text-align:center;}
#sidebar .sidebar-footer{padding:16px 24px;border-top:1px solid var(--ss-border);}
#sidebar .sidebar-footer small{color:var(--ss-muted);font-size:.72rem;}
/* Main content */
#main{margin-left:260px;flex:1;min-height:100vh;display:flex;flex-direction:column;}
.topbar{background:var(--ss-sidebar);border-bottom:1px solid var(--ss-border);padding:14px 28px;display:flex;align-items:center;justify-content:space-between;}
.topbar h6{color:var(--ss-text);font-size:.95rem;font-weight:600;margin:0;}
.topbar .badge-watsonx{background:linear-gradient(90deg,#7c5cd8,#00d4ff);color:#fff;padding:4px 12px;border-radius:20px;font-size:.7rem;font-weight:600;}
.content{padding:28px;flex:1;}
/* Cards */
.ss-card{background:var(--ss-card);border:1px solid var(--ss-border);border-radius:12px;padding:22px;transition:border-color .2s;}
.ss-card:hover{border-color:var(--ss-accent);}
.agent-icon{width:52px;height:52px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;}
.bg-agent1{background:rgba(0,212,255,.15);color:var(--ss-accent);}
.bg-agent2{background:rgba(255,75,110,.15);color:var(--ss-danger);}
.bg-agent3{background:rgba(124,92,216,.15);color:var(--ss-accent2);}
.bg-agent4{background:rgba(0,255,157,.15);color:var(--ss-accent3);}
/* Form controls */
.form-control,.form-select{background:#1a2235;border:1px solid var(--ss-border);color:var(--ss-text);border-radius:8px;}
.form-control:focus,.form-select:focus{background:#1a2235;border-color:var(--ss-accent);color:var(--ss-text);box-shadow:0 0 0 3px rgba(0,212,255,.15);}
.form-label{color:#94a3b8;font-size:.82rem;font-weight:500;margin-bottom:5px;}
.form-control::placeholder{color:var(--ss-muted);}
/* Buttons */
.btn-signal{background:linear-gradient(135deg,#00d4ff,#7c5cd8);color:#fff;border:none;border-radius:8px;font-weight:600;padding:10px 24px;}
.btn-signal:hover{opacity:.9;color:#fff;}
.btn-outline-signal{border:1px solid var(--ss-accent);color:var(--ss-accent);background:transparent;border-radius:8px;padding:10px 24px;font-weight:600;}
.btn-outline-signal:hover{background:rgba(0,212,255,.1);color:var(--ss-accent);}
/* Result area */
.result-box{background:#0d1323;border:1px solid var(--ss-border);border-radius:10px;padding:20px;min-height:120px;white-space:pre-wrap;font-size:.88rem;line-height:1.7;color:var(--ss-text);}
.result-box.loading{color:var(--ss-muted);font-style:italic;}
/* Metric badges */
.metric-badge{background:#1a2235;border:1px solid var(--ss-border);border-radius:8px;padding:12px 16px;text-align:center;}
.metric-badge .val{font-size:1.5rem;font-weight:700;color:var(--ss-accent);}
.metric-badge .lbl{font-size:.72rem;color:var(--ss-muted);margin-top:2px;}
/* Status pills */
.pill-good{background:rgba(0,255,157,.15);color:var(--ss-accent3);border-radius:20px;padding:2px 10px;font-size:.75rem;}
.pill-warn{background:rgba(255,184,0,.15);color:var(--ss-warn);border-radius:20px;padding:2px 10px;font-size:.75rem;}
.pill-bad{background:rgba(255,75,110,.15);color:var(--ss-danger);border-radius:20px;padding:2px 10px;font-size:.75rem;}
/* Section headers */
.section-title{color:var(--ss-accent);font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;}
/* Chart container */
.chart-wrap{position:relative;height:220px;}
/* Hero gradient */
.hero-banner{background:linear-gradient(135deg,#0d1323 0%,#111827 50%,#0a1628 100%);border:1px solid var(--ss-border);border-radius:16px;padding:36px 32px;position:relative;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,rgba(0,212,255,.12),transparent 70%);pointer-events:none;}
.hero-banner::after{content:'';position:absolute;bottom:-40px;left:-40px;width:180px;height:180px;background:radial-gradient(circle,rgba(124,92,216,.1),transparent 70%);pointer-events:none;}
/* Spinner */
.spinner-signal{width:20px;height:20px;border:3px solid rgba(0,212,255,.2);border-top-color:var(--ss-accent);border-radius:50%;animation:spin .7s linear infinite;display:inline-block;vertical-align:middle;margin-right:8px;}
@keyframes spin{to{transform:rotate(360deg);}}
/* About page */
.arch-step{background:#1a2235;border:1px solid var(--ss-border);border-radius:10px;padding:16px;position:relative;}
.arch-step .step-num{position:absolute;top:-12px;left:16px;background:var(--ss-accent);color:#0a0e1a;font-size:.7rem;font-weight:700;border-radius:20px;padding:2px 10px;}
/* Responsive */
@media(max-width:768px){
  #sidebar{transform:translateX(-260px);}
  #sidebar.open{transform:translateX(0);}
  #main{margin-left:0;}
}
/* Custom scrollbar */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--ss-bg);}
::-webkit-scrollbar-thumb{background:var(--ss-border);border-radius:4px;}
</style>
</head>
<body>

<!-- ===== SIDEBAR ===== -->
<div id="sidebar">
  <div class="brand">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#00d4ff,#7c5cd8);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
        <i class="bi bi-broadcast" style="color:#fff;"></i>
      </div>
      <div>
        <h5 class="mb-0">SignalSense AI</h5>
        <small>Signal Quality Analyzer</small>
      </div>
    </div>
  </div>
  <nav>
    <div style="padding:8px 24px 4px;font-size:.68rem;color:var(--ss-muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Navigation</div>
    <a href="/" class="{{ 'active' if active_page=='home' else '' }}"><i class="bi bi-house-door"></i> Dashboard Home</a>
    <a href="/analysis" class="{{ 'active' if active_page=='analysis' else '' }}"><i class="bi bi-graph-up-arrow"></i> Signal Analysis</a>
    <a href="/detection" class="{{ 'active' if active_page=='detection' else '' }}"><i class="bi bi-exclamation-triangle"></i> Issue Detection</a>
    <a href="/optimization" class="{{ 'active' if active_page=='optimization' else '' }}"><i class="bi bi-lightning-charge"></i> Optimization Advisor</a>
    <a href="/predictive" class="{{ 'active' if active_page=='predictive' else '' }}"><i class="bi bi-activity"></i> Predictive Analytics</a>
    <div style="padding:12px 24px 4px;font-size:.68rem;color:var(--ss-muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:8px;">Information</div>
    <a href="/about" class="{{ 'active' if active_page=='about' else '' }}"><i class="bi bi-info-circle"></i> About</a>
  </nav>
  <div class="sidebar-footer">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <div style="width:8px;height:8px;background:#00ff9d;border-radius:50%;animation:pulse 2s infinite;"></div>
      <small style="color:#00ff9d;font-size:.72rem;">IBM watsonx.ai Connected</small>
    </div>
    <small>Llama 3.3 70B Instruct • Multi-Agent AI</small>
  </div>
</div>

<!-- ===== MAIN ===== -->
<div id="main">
  <div class="topbar">
    <div style="display:flex;align-items:center;gap:12px;">
      <button class="btn btn-sm d-md-none" onclick="document.getElementById('sidebar').classList.toggle('open')" style="color:var(--ss-text);background:var(--ss-border);border:none;border-radius:6px;padding:4px 10px;"><i class="bi bi-list"></i></button>
      <h6>{{ page_title }}</h6>
    </div>
    <span class="badge-watsonx"><i class="bi bi-stars"></i> IBM watsonx.ai · Llama 3.3 70B</span>
  </div>
  <div class="content">
    {{ content | safe }}
  </div>
</div>

<style>@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}</style>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
function showLoading(resultId, btnId){
  document.getElementById(resultId).innerHTML = '<span class="spinner-signal"></span> Analyzing with IBM watsonx.ai Granite Model...';
  document.getElementById(resultId).className = 'result-box loading';
  if(btnId){ document.getElementById(btnId).disabled = true; }
}
function showResult(resultId, text, btnId){
  document.getElementById(resultId).innerText = text;
  document.getElementById(resultId).className = 'result-box';
  if(btnId){ document.getElementById(btnId).disabled = false; }
}
async function runAgent(endpoint, payload, resultId, btnId){
  showLoading(resultId, btnId);
  try{
    const resp = await fetch(endpoint, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    showResult(resultId, data.result || data.error || 'No response.', btnId);
  } catch(e){
    showResult(resultId, 'Error: ' + e.message, btnId);
  }
}
</script>
</body>
</html>
"""


# =============================================================================
# PAGE CONTENT TEMPLATES
# =============================================================================

HOME_CONTENT = """
<div class="hero-banner mb-4">
  <div class="row align-items-center">
    <div class="col-lg-8">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
        <span style="background:rgba(0,212,255,.15);color:#00d4ff;padding:4px 14px;border-radius:20px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;"><i class="bi bi-stars me-1"></i>Agentic AI · IBM watsonx.ai</span>
      </div>
      <h1 style="font-size:2rem;font-weight:800;color:#fff;line-height:1.25;margin-bottom:14px;">Intelligent Signal Quality<br><span style="background:linear-gradient(90deg,#00d4ff,#7c5cd8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Analysis &amp; Diagnostics</span></h1>
      <p style="color:#94a3b8;font-size:.95rem;line-height:1.7;max-width:540px;margin-bottom:22px;">Four specialized AI agents powered by IBM watsonx.ai (Llama 3.3 70B Instruct) to analyze signal quality, detect anomalies, optimize communication, and predict network failures — all in real time.</p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <a href="/analysis" class="btn btn-signal"><i class="bi bi-graph-up-arrow me-2"></i>Start Analysis</a>
        <a href="/about" class="btn btn-outline-signal"><i class="bi bi-info-circle me-2"></i>Learn More</a>
      </div>
    </div>
    <div class="col-lg-4 d-none d-lg-flex justify-content-end">
      <div style="width:180px;height:180px;background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.2);border-radius:50%;display:flex;align-items:center;justify-content:center;">
        <i class="bi bi-broadcast" style="font-size:5rem;color:rgba(0,212,255,.4);"></i>
      </div>
    </div>
  </div>
</div>

<!-- KPI Row -->
<div class="row g-3 mb-4">
  <div class="col-6 col-md-3">
    <div class="metric-badge">
      <div class="val" style="color:#00d4ff;"><i class="bi bi-cpu" style="font-size:1.2rem;"></i></div>
      <div class="lbl">4 AI Agents</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="metric-badge">
      <div class="val" style="color:#7c5cd8;"><i class="bi bi-stars" style="font-size:1.2rem;"></i></div>
      <div class="lbl">Llama 3.3 70B</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="metric-badge">
      <div class="val" style="color:#00ff9d;"><i class="bi bi-shield-check" style="font-size:1.2rem;"></i></div>
      <div class="lbl">Real-time Detection</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="metric-badge">
      <div class="val" style="color:#ffb800;"><i class="bi bi-activity" style="font-size:1.2rem;"></i></div>
      <div class="lbl">Predictive AI</div>
    </div>
  </div>
</div>

<!-- Agent Cards -->
<div class="section-title">AI Agent Suite</div>
<div class="row g-3 mb-4">
  <div class="col-md-6 col-xl-3">
    <div class="ss-card h-100">
      <div class="d-flex align-items-start gap-3 mb-3">
        <div class="agent-icon bg-agent1"><i class="bi bi-graph-up-arrow"></i></div>
        <div>
          <div style="font-weight:700;font-size:.95rem;color:#fff;margin-bottom:2px;">Signal Analysis Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 1</div>
        </div>
      </div>
      <p style="font-size:.82rem;color:#94a3b8;line-height:1.6;margin-bottom:16px;">Evaluates SNR, BER, latency, throughput, RSSI, jitter, and packet loss. Generates comprehensive signal quality reports.</p>
      <div class="d-flex flex-wrap gap-1 mb-3">
        <span style="background:rgba(0,212,255,.1);color:#00d4ff;padding:2px 8px;border-radius:12px;font-size:.68rem;">SNR Analysis</span>
        <span style="background:rgba(0,212,255,.1);color:#00d4ff;padding:2px 8px;border-radius:12px;font-size:.68rem;">BER Evaluation</span>
        <span style="background:rgba(0,212,255,.1);color:#00d4ff;padding:2px 8px;border-radius:12px;font-size:.68rem;">Performance Trends</span>
      </div>
      <a href="/analysis" class="btn btn-signal w-100" style="font-size:.82rem;padding:8px;">Analyze Signal</a>
    </div>
  </div>
  <div class="col-md-6 col-xl-3">
    <div class="ss-card h-100">
      <div class="d-flex align-items-start gap-3 mb-3">
        <div class="agent-icon bg-agent2"><i class="bi bi-exclamation-triangle"></i></div>
        <div>
          <div style="font-weight:700;font-size:.95rem;color:#fff;margin-bottom:2px;">Issue Detection Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 2</div>
        </div>
      </div>
      <p style="font-size:.82rem;color:#94a3b8;line-height:1.6;margin-bottom:16px;">Detects interference, signal degradation, congestion, jitter spikes, and communication anomalies with root cause analysis.</p>
      <div class="d-flex flex-wrap gap-1 mb-3">
        <span style="background:rgba(255,75,110,.1);color:#ff4b6e;padding:2px 8px;border-radius:12px;font-size:.68rem;">Interference</span>
        <span style="background:rgba(255,75,110,.1);color:#ff4b6e;padding:2px 8px;border-radius:12px;font-size:.68rem;">Root Cause</span>
        <span style="background:rgba(255,75,110,.1);color:#ff4b6e;padding:2px 8px;border-radius:12px;font-size:.68rem;">Severity Rating</span>
      </div>
      <a href="/detection" class="btn w-100" style="background:linear-gradient(135deg,#ff4b6e,#7c5cd8);color:#fff;border:none;border-radius:8px;font-weight:600;font-size:.82rem;padding:8px;">Detect Issues</a>
    </div>
  </div>
  <div class="col-md-6 col-xl-3">
    <div class="ss-card h-100">
      <div class="d-flex align-items-start gap-3 mb-3">
        <div class="agent-icon bg-agent3"><i class="bi bi-lightning-charge"></i></div>
        <div>
          <div style="font-weight:700;font-size:.95rem;color:#fff;margin-bottom:2px;">Optimization Advisor</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 3</div>
        </div>
      </div>
      <p style="font-size:.82rem;color:#94a3b8;line-height:1.6;margin-bottom:16px;">Generates AI-powered recommendations for antenna optimization, frequency planning, and network configuration improvements.</p>
      <div class="d-flex flex-wrap gap-1 mb-3">
        <span style="background:rgba(124,92,216,.1);color:#7c5cd8;padding:2px 8px;border-radius:12px;font-size:.68rem;">Antenna Tuning</span>
        <span style="background:rgba(124,92,216,.1);color:#7c5cd8;padding:2px 8px;border-radius:12px;font-size:.68rem;">Noise Reduction</span>
        <span style="background:rgba(124,92,216,.1);color:#7c5cd8;padding:2px 8px;border-radius:12px;font-size:.68rem;">Config Advice</span>
      </div>
      <a href="/optimization" class="btn w-100" style="background:linear-gradient(135deg,#7c5cd8,#00d4ff);color:#fff;border:none;border-radius:8px;font-weight:600;font-size:.82rem;padding:8px;">Get Recommendations</a>
    </div>
  </div>
  <div class="col-md-6 col-xl-3">
    <div class="ss-card h-100">
      <div class="d-flex align-items-start gap-3 mb-3">
        <div class="agent-icon bg-agent4"><i class="bi bi-activity"></i></div>
        <div>
          <div style="font-weight:700;font-size:.95rem;color:#fff;margin-bottom:2px;">Predictive Failure Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 4</div>
        </div>
      </div>
      <p style="font-size:.82rem;color:#94a3b8;line-height:1.6;margin-bottom:16px;">Forecasts signal degradation, estimates failure probabilities, and schedules preventive maintenance using historical data.</p>
      <div class="d-flex flex-wrap gap-1 mb-3">
        <span style="background:rgba(0,255,157,.1);color:#00ff9d;padding:2px 8px;border-radius:12px;font-size:.68rem;">Failure Forecast</span>
        <span style="background:rgba(0,255,157,.1);color:#00ff9d;padding:2px 8px;border-radius:12px;font-size:.68rem;">Risk Assessment</span>
        <span style="background:rgba(0,255,157,.1);color:#00ff9d;padding:2px 8px;border-radius:12px;font-size:.68rem;">Maintenance Plan</span>
      </div>
      <a href="/predictive" class="btn w-100" style="background:linear-gradient(135deg,#00ff9d,#00d4ff);color:#0a0e1a;border:none;border-radius:8px;font-weight:700;font-size:.82rem;padding:8px;">Predict Failures</a>
    </div>
  </div>
</div>

<!-- Charts Row -->
<div class="section-title">Live Signal Metrics Preview</div>
<div class="row g-3">
  <div class="col-md-6">
    <div class="ss-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div style="font-size:.88rem;font-weight:600;color:#fff;">SNR Trend (dB)</div>
        <span class="pill-good">Good</span>
      </div>
      <div class="chart-wrap"><canvas id="snrChart"></canvas></div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="ss-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div style="font-size:.88rem;font-weight:600;color:#fff;">Latency &amp; Jitter (ms)</div>
        <span class="pill-warn">Monitor</span>
      </div>
      <div class="chart-wrap"><canvas id="latChart"></canvas></div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="ss-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div style="font-size:.88rem;font-weight:600;color:#fff;">Packet Loss &amp; BER</div>
        <span class="pill-bad">High</span>
      </div>
      <div class="chart-wrap"><canvas id="plChart"></canvas></div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="ss-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div style="font-size:.88rem;font-weight:600;color:#fff;">Throughput (Mbps)</div>
        <span class="pill-good">Stable</span>
      </div>
      <div class="chart-wrap"><canvas id="tpChart"></canvas></div>
    </div>
  </div>
</div>

<script>
const labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const chartOpts = (color) => ({
  responsive:true,maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#64748b',font:{size:11}}},tooltip:{backgroundColor:'#111827',titleColor:'#e2e8f0',bodyColor:'#94a3b8'}},
  scales:{x:{grid:{color:'#1e2d45'},ticks:{color:'#64748b',font:{size:10}}},y:{grid:{color:'#1e2d45'},ticks:{color:'#64748b',font:{size:10}}}}
});
new Chart('snrChart',{type:'line',data:{labels,datasets:[{label:'SNR (dB)',data:[32,35,28,40,38,36,42],borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.08)',tension:.4,pointRadius:3,fill:true}]},options:chartOpts('#00d4ff')});
new Chart('latChart',{type:'line',data:{labels,datasets:[{label:'Latency',data:[22,28,35,30,18,24,20],borderColor:'#ffb800',backgroundColor:'rgba(255,184,0,.06)',tension:.4,pointRadius:3,fill:true},{label:'Jitter',data:[5,8,12,7,4,6,5],borderColor:'#7c5cd8',backgroundColor:'rgba(124,92,216,.06)',tension:.4,pointRadius:3,fill:true}]},options:chartOpts()});
new Chart('plChart',{type:'bar',data:{labels,datasets:[{label:'Packet Loss %',data:[0.8,1.2,2.5,1.8,0.6,1.0,0.9],backgroundColor:'rgba(255,75,110,.5)',borderColor:'#ff4b6e',borderWidth:1}]},options:chartOpts()});
new Chart('tpChart',{type:'line',data:{labels,datasets:[{label:'Throughput',data:[95,88,72,85,98,92,96],borderColor:'#00ff9d',backgroundColor:'rgba(0,255,157,.08)',tension:.4,pointRadius:3,fill:true}]},options:chartOpts()});
</script>
"""

ANALYSIS_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ss-card">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <div class="agent-icon bg-agent1"><i class="bi bi-graph-up-arrow"></i></div>
        <div>
          <div style="font-weight:700;color:#fff;">Signal Analysis Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 1 • IBM Granite Model</div>
        </div>
      </div>
      <form id="analysisForm">
        <div class="row g-3">
          <div class="col-6">
            <label class="form-label">SNR (dB)</label>
            <input class="form-control" id="a_snr" placeholder="e.g. 35" type="number" step="0.1"/>
          </div>
          <div class="col-6">
            <label class="form-label">BER</label>
            <input class="form-control" id="a_ber" placeholder="e.g. 0.001" type="text"/>
          </div>
          <div class="col-6">
            <label class="form-label">Latency (ms)</label>
            <input class="form-control" id="a_latency" placeholder="e.g. 20" type="number"/>
          </div>
          <div class="col-6">
            <label class="form-label">Throughput (Mbps)</label>
            <input class="form-control" id="a_throughput" placeholder="e.g. 85" type="number"/>
          </div>
          <div class="col-6">
            <label class="form-label">Packet Loss (%)</label>
            <input class="form-control" id="a_packet_loss" placeholder="e.g. 0.5" type="number" step="0.01"/>
          </div>
          <div class="col-6">
            <label class="form-label">RSSI (dBm)</label>
            <input class="form-control" id="a_rssi" placeholder="e.g. -65" type="number"/>
          </div>
          <div class="col-6">
            <label class="form-label">Jitter (ms)</label>
            <input class="form-control" id="a_jitter" placeholder="e.g. 4" type="number" step="0.1"/>
          </div>
          <div class="col-6">
            <label class="form-label">Frequency Band</label>
            <input class="form-control" id="a_frequency_band" placeholder="e.g. 2.4 GHz"/>
          </div>
          <div class="col-12">
            <label class="form-label">Historical Observations</label>
            <textarea class="form-control" id="a_historical" rows="2" placeholder="Paste historical readings or describe trends..."></textarea>
          </div>
          <div class="col-12">
            <label class="form-label">Additional Notes</label>
            <textarea class="form-control" id="a_notes" rows="2" placeholder="Describe the deployment environment or any additional context..."></textarea>
          </div>
          <div class="col-12">
            <label class="form-label"><i class="bi bi-file-earmark-spreadsheet me-1"></i>Upload CSV Dataset (optional)</label>
            <input type="file" class="form-control" id="a_csv" accept=".csv"/>
            <div style="font-size:.72rem;color:var(--ss-muted);margin-top:4px;">Columns: snr,ber,latency,throughput,packet_loss,rssi,jitter</div>
          </div>
        </div>
        <div class="mt-3 d-flex gap-2">
          <button type="button" class="btn btn-signal flex-fill" id="analyzeBtn" onclick="submitAnalysis()">
            <i class="bi bi-stars me-2"></i>Analyze with AI
          </button>
          <button type="button" class="btn btn-outline-signal" onclick="loadSampleAnalysis()">
            <i class="bi bi-file-text"></i> Sample
          </button>
        </div>
      </form>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ss-card" style="height:100%;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div class="section-title mb-0">AI Analysis Report</div>
        <span style="font-size:.72rem;color:var(--ss-muted);">Powered by IBM Granite</span>
      </div>
      <div class="result-box loading" id="analysisResult">Enter signal parameters and click <strong>Analyze with AI</strong> to generate an IBM Granite-powered signal quality report.</div>
      <div class="row g-2 mt-3" id="metricPreview" style="display:none!important;">
        <div class="col-4"><div class="metric-badge"><div class="val" id="pv_snr" style="font-size:1.1rem;">–</div><div class="lbl">SNR (dB)</div></div></div>
        <div class="col-4"><div class="metric-badge"><div class="val" id="pv_lat" style="font-size:1.1rem;color:#ffb800;">–</div><div class="lbl">Latency (ms)</div></div></div>
        <div class="col-4"><div class="metric-badge"><div class="val" id="pv_pl" style="font-size:1.1rem;color:#ff4b6e;">–</div><div class="lbl">Pkt Loss %</div></div></div>
      </div>
    </div>
  </div>
</div>
<script>
function loadSampleAnalysis(){
  document.getElementById('a_snr').value=32;
  document.getElementById('a_ber').value='0.002';
  document.getElementById('a_latency').value=45;
  document.getElementById('a_throughput').value=72;
  document.getElementById('a_packet_loss').value=1.8;
  document.getElementById('a_rssi').value=-72;
  document.getElementById('a_jitter').value=8;
  document.getElementById('a_frequency_band').value='5 GHz';
  document.getElementById('a_historical').value='Last week: SNR ranged 28-36 dB, latency averaging 40ms. Notable degradation Thursday-Friday.';
  document.getElementById('a_notes').value='Outdoor campus WiFi deployment. High user density during peak hours.';
}
async function submitAnalysis(){
  const csvFile = document.getElementById('a_csv').files[0];
  let csvText = '';
  if(csvFile){
    csvText = await csvFile.text();
  }
  const payload = {
    snr: document.getElementById('a_snr').value,
    ber: document.getElementById('a_ber').value,
    latency: document.getElementById('a_latency').value,
    throughput: document.getElementById('a_throughput').value,
    packet_loss: document.getElementById('a_packet_loss').value,
    rssi: document.getElementById('a_rssi').value,
    jitter: document.getElementById('a_jitter').value,
    frequency_band: document.getElementById('a_frequency_band').value,
    historical: document.getElementById('a_historical').value,
    notes: document.getElementById('a_notes').value,
    csv_data: csvText
  };
  showLoading('analysisResult','analyzeBtn');
  try{
    const resp = await fetch('/api/analysis',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await resp.json();
    showResult('analysisResult', data.result || data.error || 'No response.', 'analyzeBtn');
    document.getElementById('metricPreview').style.display='flex';
    document.getElementById('pv_snr').innerText = payload.snr || '–';
    document.getElementById('pv_lat').innerText = payload.latency || '–';
    document.getElementById('pv_pl').innerText = payload.packet_loss || '–';
  }catch(e){showResult('analysisResult','Error: '+e.message,'analyzeBtn');}
}
</script>
"""

DETECTION_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ss-card">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <div class="agent-icon bg-agent2"><i class="bi bi-exclamation-triangle"></i></div>
        <div>
          <div style="font-weight:700;color:#fff;">Signal Issue Detection Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 2 • IBM Granite Model</div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-6"><label class="form-label">SNR (dB)</label><input class="form-control" id="d_snr" placeholder="e.g. 18" type="number" step="0.1"/></div>
        <div class="col-6"><label class="form-label">BER</label><input class="form-control" id="d_ber" placeholder="e.g. 0.05" type="text"/></div>
        <div class="col-6"><label class="form-label">Latency (ms)</label><input class="form-control" id="d_latency" placeholder="e.g. 180" type="number"/></div>
        <div class="col-6"><label class="form-label">Packet Loss (%)</label><input class="form-control" id="d_packet_loss" placeholder="e.g. 5.2" type="number" step="0.01"/></div>
        <div class="col-6"><label class="form-label">RSSI (dBm)</label><input class="form-control" id="d_rssi" placeholder="e.g. -88" type="number"/></div>
        <div class="col-6"><label class="form-label">Jitter (ms)</label><input class="form-control" id="d_jitter" placeholder="e.g. 22" type="number" step="0.1"/></div>
        <div class="col-6"><label class="form-label">Frequency Band</label><input class="form-control" id="d_frequency_band" placeholder="e.g. 900 MHz"/></div>
        <div class="col-6">
          <label class="form-label">Environment</label>
          <select class="form-select" id="d_environment">
            <option value="">Select environment</option>
            <option>Urban Dense</option><option>Suburban</option><option>Rural</option>
            <option>Industrial</option><option>Indoor Office</option><option>Outdoor Campus</option>
          </select>
        </div>
        <div class="col-12">
          <label class="form-label">Reported Symptoms / Observations</label>
          <textarea class="form-control" id="d_symptoms" rows="3" placeholder="Describe symptoms: dropped calls, slow speeds, intermittent connectivity..."></textarea>
        </div>
      </div>
      <div class="mt-3 d-flex gap-2">
        <button class="btn btn-signal flex-fill" style="background:linear-gradient(135deg,#ff4b6e,#7c5cd8);" id="detectBtn" onclick="submitDetection()">
          <i class="bi bi-search me-2"></i>Detect Issues
        </button>
        <button class="btn btn-outline-signal" onclick="loadSampleDetection()"><i class="bi bi-file-text"></i> Sample</button>
      </div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ss-card h-100">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div class="section-title mb-0" style="color:#ff4b6e;">Issue Detection Report</div>
        <span style="font-size:.72rem;color:var(--ss-muted);">Powered by IBM Granite</span>
      </div>
      <div class="result-box loading" id="detectionResult">Enter signal parameters and click <strong>Detect Issues</strong> to identify communication problems with IBM Granite.</div>
      <div class="row g-2 mt-3">
        <div class="col-12">
          <div style="background:#1a2235;border:1px solid var(--ss-border);border-radius:8px;padding:14px;">
            <div style="font-size:.75rem;color:var(--ss-muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;">Common Issue Checklist</div>
            <div class="row g-2">
              <div class="col-6"><span style="font-size:.78rem;color:#94a3b8;"><i class="bi bi-dash-circle me-1" style="color:#ff4b6e;"></i>Low SNR (&lt;15dB)</span></div>
              <div class="col-6"><span style="font-size:.78rem;color:#94a3b8;"><i class="bi bi-dash-circle me-1" style="color:#ff4b6e;"></i>High BER (&gt;1e-3)</span></div>
              <div class="col-6"><span style="font-size:.78rem;color:#94a3b8;"><i class="bi bi-dash-circle me-1" style="color:#ffb800;"></i>High Latency (&gt;100ms)</span></div>
              <div class="col-6"><span style="font-size:.78rem;color:#94a3b8;"><i class="bi bi-dash-circle me-1" style="color:#ffb800;"></i>Packet Loss (&gt;2%)</span></div>
              <div class="col-6"><span style="font-size:.78rem;color:#94a3b8;"><i class="bi bi-dash-circle me-1" style="color:#ff4b6e;"></i>Weak RSSI (&lt;-80dBm)</span></div>
              <div class="col-6"><span style="font-size:.78rem;color:#94a3b8;"><i class="bi bi-dash-circle me-1" style="color:#ffb800;"></i>High Jitter (&gt;10ms)</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
function loadSampleDetection(){
  document.getElementById('d_snr').value=14;
  document.getElementById('d_ber').value='0.048';
  document.getElementById('d_latency').value=195;
  document.getElementById('d_packet_loss').value=6.5;
  document.getElementById('d_rssi').value=-89;
  document.getElementById('d_jitter').value=25;
  document.getElementById('d_frequency_band').value='2.4 GHz';
  document.getElementById('d_environment').value='Industrial';
  document.getElementById('d_symptoms').value='Frequent dropped packets, unstable connection, interference suspected near industrial equipment. Users report slow data rates and intermittent disconnections.';
}
async function submitDetection(){
  const payload={
    snr:document.getElementById('d_snr').value,
    ber:document.getElementById('d_ber').value,
    latency:document.getElementById('d_latency').value,
    packet_loss:document.getElementById('d_packet_loss').value,
    rssi:document.getElementById('d_rssi').value,
    jitter:document.getElementById('d_jitter').value,
    frequency_band:document.getElementById('d_frequency_band').value,
    environment:document.getElementById('d_environment').value,
    symptoms:document.getElementById('d_symptoms').value
  };
  showLoading('detectionResult','detectBtn');
  try{
    const resp=await fetch('/api/detection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await resp.json();
    showResult('detectionResult',data.result||data.error||'No response.','detectBtn');
  }catch(e){showResult('detectionResult','Error: '+e.message,'detectBtn');}
}
</script>
"""

OPTIMIZATION_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ss-card">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <div class="agent-icon bg-agent3"><i class="bi bi-lightning-charge"></i></div>
        <div>
          <div style="font-weight:700;color:#fff;">Optimization Advisor Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 3 • IBM Granite Model</div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-6"><label class="form-label">SNR (dB)</label><input class="form-control" id="o_snr" placeholder="e.g. 22" type="number" step="0.1"/></div>
        <div class="col-6"><label class="form-label">BER</label><input class="form-control" id="o_ber" placeholder="e.g. 0.01" type="text"/></div>
        <div class="col-6"><label class="form-label">Latency (ms)</label><input class="form-control" id="o_latency" placeholder="e.g. 80" type="number"/></div>
        <div class="col-6"><label class="form-label">Throughput (Mbps)</label><input class="form-control" id="o_throughput" placeholder="e.g. 50" type="number"/></div>
        <div class="col-6"><label class="form-label">Packet Loss (%)</label><input class="form-control" id="o_packet_loss" placeholder="e.g. 2.0" type="number" step="0.01"/></div>
        <div class="col-6"><label class="form-label">RSSI (dBm)</label><input class="form-control" id="o_rssi" placeholder="e.g. -75" type="number"/></div>
        <div class="col-6"><label class="form-label">Jitter (ms)</label><input class="form-control" id="o_jitter" placeholder="e.g. 10" type="number" step="0.1"/></div>
        <div class="col-6">
          <label class="form-label">Network Type</label>
          <select class="form-select" id="o_network_type">
            <option value="">Select type</option>
            <option>WiFi 802.11ac</option><option>WiFi 802.11ax (WiFi 6)</option>
            <option>4G LTE</option><option>5G NR</option><option>Fiber Optic</option>
            <option>Microwave Backhaul</option><option>Satellite</option><option>LoRa/IoT</option>
          </select>
        </div>
        <div class="col-12">
          <label class="form-label">Frequency Band</label>
          <input class="form-control" id="o_frequency_band" placeholder="e.g. 5 GHz, Band 7"/>
        </div>
        <div class="col-12">
          <label class="form-label">Communication Environment</label>
          <textarea class="form-control" id="o_environment" rows="2" placeholder="Describe deployment: building type, obstructions, user density..."></textarea>
        </div>
        <div class="col-12">
          <label class="form-label">Application Requirements</label>
          <textarea class="form-control" id="o_app_requirements" rows="2" placeholder="e.g. VoIP, video streaming, IoT sensors, industrial automation..."></textarea>
        </div>
        <div class="col-12">
          <label class="form-label">Identified Problems (optional)</label>
          <textarea class="form-control" id="o_problems" rows="2" placeholder="Describe existing issues you want to address..."></textarea>
        </div>
      </div>
      <div class="mt-3 d-flex gap-2">
        <button class="btn flex-fill" style="background:linear-gradient(135deg,#7c5cd8,#00d4ff);color:#fff;border:none;border-radius:8px;font-weight:600;" id="optBtn" onclick="submitOptimization()">
          <i class="bi bi-lightning-charge-fill me-2"></i>Generate Recommendations
        </button>
        <button class="btn btn-outline-signal" onclick="loadSampleOptimization()"><i class="bi bi-file-text"></i> Sample</button>
      </div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ss-card h-100">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div class="section-title mb-0" style="color:#7c5cd8;">AI Optimization Recommendations</div>
        <span style="font-size:.72rem;color:var(--ss-muted);">Powered by IBM Granite</span>
      </div>
      <div class="result-box loading" id="optResult">Fill in signal parameters and network context, then click <strong>Generate Recommendations</strong> for AI-powered engineering advice.</div>
      <div class="mt-3">
        <div style="background:#1a2235;border:1px dashed var(--ss-border);border-radius:8px;padding:14px;">
          <div style="font-size:.75rem;color:var(--ss-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Optimization Categories</div>
          <div class="row g-2">
            <div class="col-6 col-md-4"><div style="font-size:.78rem;color:#7c5cd8;"><i class="bi bi-broadcast-pin me-1"></i>Antenna Tuning</div></div>
            <div class="col-6 col-md-4"><div style="font-size:.78rem;color:#7c5cd8;"><i class="bi bi-soundwave me-1"></i>Noise Reduction</div></div>
            <div class="col-6 col-md-4"><div style="font-size:.78rem;color:#7c5cd8;"><i class="bi bi-diagram-3 me-1"></i>Freq Planning</div></div>
            <div class="col-6 col-md-4"><div style="font-size:.78rem;color:#7c5cd8;"><i class="bi bi-gear me-1"></i>Config Tuning</div></div>
            <div class="col-6 col-md-4"><div style="font-size:.78rem;color:#7c5cd8;"><i class="bi bi-shield-check me-1"></i>Reliability</div></div>
            <div class="col-6 col-md-4"><div style="font-size:.78rem;color:#7c5cd8;"><i class="bi bi-tools me-1"></i>Equipment</div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
function loadSampleOptimization(){
  document.getElementById('o_snr').value=20;
  document.getElementById('o_ber').value='0.012';
  document.getElementById('o_latency').value=85;
  document.getElementById('o_throughput').value=48;
  document.getElementById('o_packet_loss').value=2.2;
  document.getElementById('o_rssi').value=-76;
  document.getElementById('o_jitter').value=12;
  document.getElementById('o_network_type').value='WiFi 802.11ac';
  document.getElementById('o_frequency_band').value='2.4 GHz';
  document.getElementById('o_environment').value='Large open-plan office, 3 floors, concrete walls, 200+ users.';
  document.getElementById('o_app_requirements').value='VoIP calls, video conferencing, cloud applications. Low latency critical.';
  document.getElementById('o_problems').value='Users report poor call quality and video lag during peak hours. SNR drops near elevator shafts.';
}
async function submitOptimization(){
  const payload={
    snr:document.getElementById('o_snr').value,
    ber:document.getElementById('o_ber').value,
    latency:document.getElementById('o_latency').value,
    throughput:document.getElementById('o_throughput').value,
    packet_loss:document.getElementById('o_packet_loss').value,
    rssi:document.getElementById('o_rssi').value,
    jitter:document.getElementById('o_jitter').value,
    network_type:document.getElementById('o_network_type').value,
    frequency_band:document.getElementById('o_frequency_band').value,
    environment:document.getElementById('o_environment').value,
    app_requirements:document.getElementById('o_app_requirements').value,
    problems:document.getElementById('o_problems').value
  };
  showLoading('optResult','optBtn');
  try{
    const resp=await fetch('/api/optimization',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await resp.json();
    showResult('optResult',data.result||data.error||'No response.','optBtn');
  }catch(e){showResult('optResult','Error: '+e.message,'optBtn');}
}
</script>
"""

PREDICTIVE_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="ss-card">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <div class="agent-icon bg-agent4"><i class="bi bi-activity"></i></div>
        <div>
          <div style="font-weight:700;color:#fff;">Predictive Failure Agent</div>
          <div style="font-size:.75rem;color:var(--ss-muted);">Agent 4 • IBM Granite Model</div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-6"><label class="form-label">Current SNR (dB)</label><input class="form-control" id="p_snr" placeholder="e.g. 25" type="number" step="0.1"/></div>
        <div class="col-6"><label class="form-label">Current BER</label><input class="form-control" id="p_ber" placeholder="e.g. 0.003" type="text"/></div>
        <div class="col-6"><label class="form-label">Latency Trend</label>
          <select class="form-select" id="p_latency_trend">
            <option value="">Select trend</option>
            <option>Stable</option><option>Gradually Increasing</option>
            <option>Rapidly Increasing</option><option>Fluctuating</option><option>Decreasing</option>
          </select>
        </div>
        <div class="col-6"><label class="form-label">Throughput Trend</label>
          <select class="form-select" id="p_throughput_trend">
            <option value="">Select trend</option>
            <option>Stable</option><option>Gradually Decreasing</option>
            <option>Rapidly Decreasing</option><option>Fluctuating</option><option>Increasing</option>
          </select>
        </div>
        <div class="col-6"><label class="form-label">Packet Loss Trend</label>
          <select class="form-select" id="p_packet_loss_trend">
            <option value="">Select trend</option>
            <option>Stable &lt;1%</option><option>Gradually Increasing</option>
            <option>Rapidly Increasing</option><option>Intermittent Spikes</option>
          </select>
        </div>
        <div class="col-6"><label class="form-label">Equipment Age</label>
          <select class="form-select" id="p_equipment_age">
            <option value="">Select age</option>
            <option>&lt;1 year</option><option>1–3 years</option>
            <option>3–5 years</option><option>5–8 years</option><option>&gt;8 years</option>
          </select>
        </div>
        <div class="col-12"><label class="form-label">Environmental Conditions</label>
          <select class="form-select" id="p_environmental_conditions">
            <option value="">Select conditions</option>
            <option>Normal Indoor</option><option>Outdoor — Mild</option>
            <option>Outdoor — Harsh (heat/cold)</option><option>High Humidity</option>
            <option>Industrial (dust/vibration)</option><option>Coastal (salt air)</option>
          </select>
        </div>
        <div class="col-12"><label class="form-label">Historical Signal Data (7–30 days)</label>
          <textarea class="form-control" id="p_historical_data" rows="4" placeholder="Paste measurements: Date | SNR | BER | Latency | Throughput&#10;2024-01-01 | 38 | 0.001 | 22 | 95&#10;2024-01-02 | 35 | 0.002 | 28 | 88"></textarea>
        </div>
        <div class="col-12"><label class="form-label">Recent Incidents</label>
          <textarea class="form-control" id="p_recent_incidents" rows="2" placeholder="Describe any recent outages, alarms, or performance events..."></textarea>
        </div>
        <div class="col-12"><label class="form-label">Maintenance History</label>
          <textarea class="form-control" id="p_maintenance_history" rows="2" placeholder="Last maintenance date, work performed, replacement history..."></textarea>
        </div>
        <div class="col-12">
          <label class="form-label"><i class="bi bi-file-earmark-spreadsheet me-1"></i>Upload Historical CSV (optional)</label>
          <input type="file" class="form-control" id="p_csv" accept=".csv"/>
        </div>
      </div>
      <div class="mt-3 d-flex gap-2">
        <button class="btn flex-fill" style="background:linear-gradient(135deg,#00ff9d,#00d4ff);color:#0a0e1a;border:none;border-radius:8px;font-weight:700;" id="predBtn" onclick="submitPredictive()">
          <i class="bi bi-activity me-2"></i>Predict Failures
        </button>
        <button class="btn btn-outline-signal" onclick="loadSamplePredictive()"><i class="bi bi-file-text"></i> Sample</button>
      </div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="ss-card h-100">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div class="section-title mb-0" style="color:#00ff9d;">Predictive Failure Report</div>
        <span style="font-size:.72rem;color:var(--ss-muted);">Powered by IBM Granite</span>
      </div>
      <div class="result-box loading" id="predictiveResult">Provide historical signal data and click <strong>Predict Failures</strong> to receive AI-powered predictive analytics and maintenance recommendations.</div>
      <div class="mt-3">
        <div class="row g-2">
          <div class="col-12">
            <div style="background:#1a2235;border:1px solid var(--ss-border);border-radius:8px;padding:14px;">
              <div style="font-size:.75rem;color:var(--ss-muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;">Forecast Horizon</div>
              <div class="row g-2 text-center">
                <div class="col-4"><div style="font-size:.8rem;color:#00ff9d;font-weight:700;">7 Days</div><div style="font-size:.7rem;color:var(--ss-muted);">Short-term</div></div>
                <div class="col-4"><div style="font-size:.8rem;color:#ffb800;font-weight:700;">30 Days</div><div style="font-size:.7rem;color:var(--ss-muted);">Mid-term</div></div>
                <div class="col-4"><div style="font-size:.8rem;color:#ff4b6e;font-weight:700;">90 Days</div><div style="font-size:.7rem;color:var(--ss-muted);">Long-term</div></div>
              </div>
            </div>
          </div>
          <div class="col-12">
            <div class="chart-wrap" style="height:180px;"><canvas id="predChart"></canvas></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
// Illustrative predictive trend chart
new Chart('predChart',{
  type:'line',
  data:{
    labels:['Day 1','Day 7','Day 14','Day 21','Day 30','Day 60','Day 90'],
    datasets:[
      {label:'SNR Forecast (dB)',data:[32,30,27,25,22,18,14],borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.06)',tension:.4,pointRadius:3,fill:true,borderDash:[5,3]},
      {label:'Failure Risk %',data:[5,10,18,28,42,65,82],borderColor:'#ff4b6e',backgroundColor:'rgba(255,75,110,.06)',tension:.4,pointRadius:3,fill:true,borderDash:[5,3],yAxisID:'y1'}
    ]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#64748b',font:{size:10}}},tooltip:{backgroundColor:'#111827',titleColor:'#e2e8f0',bodyColor:'#94a3b8'}},
    scales:{
      x:{grid:{color:'#1e2d45'},ticks:{color:'#64748b',font:{size:9}}},
      y:{grid:{color:'#1e2d45'},ticks:{color:'#64748b',font:{size:9}},position:'left'},
      y1:{grid:{display:false},ticks:{color:'#ff4b6e',font:{size:9}},position:'right',max:100}
    }
  }
});
function loadSamplePredictive(){
  document.getElementById('p_snr').value=25;
  document.getElementById('p_ber').value='0.004';
  document.getElementById('p_latency_trend').value='Gradually Increasing';
  document.getElementById('p_throughput_trend').value='Gradually Decreasing';
  document.getElementById('p_packet_loss_trend').value='Gradually Increasing';
  document.getElementById('p_equipment_age').value='5–8 years';
  document.getElementById('p_environmental_conditions').value='Outdoor — Harsh (heat/cold)';
  document.getElementById('p_historical_data').value=`2024-01-01 | SNR:38 | BER:0.001 | Latency:22ms | Throughput:95Mbps
2024-01-07 | SNR:36 | BER:0.0015| Latency:28ms | Throughput:90Mbps
2024-01-14 | SNR:33 | BER:0.002 | Latency:35ms | Throughput:82Mbps
2024-01-21 | SNR:30 | BER:0.003 | Latency:44ms | Throughput:74Mbps
2024-01-28 | SNR:27 | BER:0.004 | Latency:55ms | Throughput:64Mbps`;
  document.getElementById('p_recent_incidents').value='Two brief outages in January. Packet loss spikes observed during heavy rain.';
  document.getElementById('p_maintenance_history').value='Last maintenance: 18 months ago. Antenna connectors replaced 2 years ago.';
}
async function submitPredictive(){
  const csvFile=document.getElementById('p_csv').files[0];
  let csvText='';
  if(csvFile){ csvText=await csvFile.text(); }
  const payload={
    snr:document.getElementById('p_snr').value,
    ber:document.getElementById('p_ber').value,
    latency_trend:document.getElementById('p_latency_trend').value,
    throughput_trend:document.getElementById('p_throughput_trend').value,
    packet_loss_trend:document.getElementById('p_packet_loss_trend').value,
    equipment_age:document.getElementById('p_equipment_age').value,
    environmental_conditions:document.getElementById('p_environmental_conditions').value,
    historical_data:document.getElementById('p_historical_data').value,
    recent_incidents:document.getElementById('p_recent_incidents').value,
    maintenance_history:document.getElementById('p_maintenance_history').value,
    csv_data:csvText
  };
  showLoading('predictiveResult','predBtn');
  try{
    const resp=await fetch('/api/predictive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await resp.json();
    showResult('predictiveResult',data.result||data.error||'No response.','predBtn');
  }catch(e){showResult('predictiveResult','Error: '+e.message,'predBtn');}
}
</script>
"""

ABOUT_CONTENT = """
<div class="row g-4">
  <div class="col-12">
    <div class="hero-banner" style="padding:28px 28px;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
        <div style="width:48px;height:48px;background:linear-gradient(135deg,#00d4ff,#7c5cd8);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;"><i class="bi bi-broadcast" style="color:#fff;"></i></div>
        <div>
          <div style="font-size:1.3rem;font-weight:800;color:#fff;">SignalSense AI</div>
          <div style="font-size:.8rem;color:var(--ss-muted);">Intelligent Signal Quality Analyzer · Powered by IBM watsonx.ai</div>
        </div>
      </div>
      <p style="color:#94a3b8;font-size:.9rem;line-height:1.7;max-width:750px;">SignalSense AI is an agentic AI platform for telecommunications and signal processing professionals. It leverages IBM watsonx.ai (Llama 3.3 70B Instruct) to deliver intelligent, context-aware analysis of signal quality, detection of communication anomalies, engineering optimization, and predictive maintenance insights.</p>
    </div>
  </div>

  <div class="col-12">
    <div class="section-title">Four-Agent Architecture</div>
    <div class="row g-3">
      <div class="col-md-6 col-xl-3">
        <div class="arch-step ss-card">
          <div class="step-num">Agent 1</div>
          <div style="margin-top:8px;display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div class="agent-icon bg-agent1" style="width:38px;height:38px;font-size:1rem;"><i class="bi bi-graph-up-arrow"></i></div>
            <div style="font-weight:700;color:#fff;font-size:.9rem;">Signal Analysis</div>
          </div>
          <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;">Evaluates SNR, BER, latency, throughput, RSSI, jitter, and packet loss. Generates comprehensive signal quality summaries and performance trend analysis using IBM Granite.</p>
        </div>
      </div>
      <div class="col-md-6 col-xl-3">
        <div class="arch-step ss-card">
          <div class="step-num" style="background:#ff4b6e;">Agent 2</div>
          <div style="margin-top:8px;display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div class="agent-icon bg-agent2" style="width:38px;height:38px;font-size:1rem;"><i class="bi bi-exclamation-triangle"></i></div>
            <div style="font-weight:700;color:#fff;font-size:.9rem;">Issue Detection</div>
          </div>
          <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;">Detects interference, attenuation, congestion, jitter spikes, and signal instability. Provides root cause analysis and severity classification with AI-powered diagnostic reasoning.</p>
        </div>
      </div>
      <div class="col-md-6 col-xl-3">
        <div class="arch-step ss-card">
          <div class="step-num" style="background:#7c5cd8;">Agent 3</div>
          <div style="margin-top:8px;display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div class="agent-icon bg-agent3" style="width:38px;height:38px;font-size:1rem;"><i class="bi bi-lightning-charge"></i></div>
            <div style="font-weight:700;color:#fff;font-size:.9rem;">Optimization Advisor</div>
          </div>
          <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;">Generates engineering recommendations: antenna alignment, noise reduction, frequency planning, modulation optimization, and network configuration improvements.</p>
        </div>
      </div>
      <div class="col-md-6 col-xl-3">
        <div class="arch-step ss-card">
          <div class="step-num" style="background:#00ff9d;color:#0a0e1a;">Agent 4</div>
          <div style="margin-top:8px;display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div class="agent-icon bg-agent4" style="width:38px;height:38px;font-size:1rem;"><i class="bi bi-activity"></i></div>
            <div style="font-weight:700;color:#fff;font-size:.9rem;">Predictive Analytics</div>
          </div>
          <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;">Analyzes historical signal trends to forecast degradation, estimate failure probabilities, and generate preventive maintenance schedules with 7/30/90-day outlooks.</p>
        </div>
      </div>
    </div>
  </div>

  <div class="col-md-6">
    <div class="ss-card h-100">
      <div class="section-title">IBM watsonx.ai Integration</div>
      <div style="display:flex;flex-direction:column;gap:12px;">
        <div style="background:#1a2235;border-radius:8px;padding:14px;">
          <div style="font-size:.82rem;font-weight:600;color:#fff;margin-bottom:6px;"><i class="bi bi-stars me-2" style="color:#7c5cd8;"></i>meta-llama/llama-3-3-70b-instruct</div>
          <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin:0;">Foundation models from IBM, optimized for enterprise AI applications including natural language understanding, technical reasoning, and domain-specific analysis.</p>
        </div>
        <div style="background:#1a2235;border-radius:8px;padding:14px;">
          <div style="font-size:.82rem;font-weight:600;color:#fff;margin-bottom:6px;"><i class="bi bi-cloud-fill me-2" style="color:#00d4ff;"></i>watsonx.ai Platform</div>
          <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin:0;">IBM's enterprise AI platform providing scalable model inference, governance, and integration capabilities for production AI deployments.</p>
        </div>
        <div style="background:#1a2235;border-radius:8px;padding:14px;">
          <div style="font-size:.82rem;font-weight:600;color:#fff;margin-bottom:6px;"><i class="bi bi-key me-2" style="color:#ffb800;"></i>Environment Variables</div>
          <div style="font-family:monospace;font-size:.75rem;color:#94a3b8;line-height:1.9;">
            WATSONX_API_KEY<br/>WATSONX_PROJECT_ID<br/>WATSONX_URL
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="col-md-6">
    <div class="ss-card h-100">
      <div class="section-title">Agent Orchestration Flow</div>
      <div style="display:flex;flex-direction:column;gap:8px;">
        <div style="background:#1a2235;border-radius:8px;padding:12px;display:flex;align-items:center;gap:12px;">
          <div style="width:32px;height:32px;background:rgba(0,212,255,.15);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#00d4ff;font-size:.85rem;font-weight:700;flex-shrink:0;">1</div>
          <div style="font-size:.8rem;color:#94a3b8;"><span style="color:#fff;font-weight:600;">User Input</span> — Parameters entered or CSV uploaded via the web interface.</div>
        </div>
        <div style="background:#1a2235;border-radius:8px;padding:12px;display:flex;align-items:center;gap:12px;">
          <div style="width:32px;height:32px;background:rgba(124,92,216,.15);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#7c5cd8;font-size:.85rem;font-weight:700;flex-shrink:0;">2</div>
          <div style="font-size:.8rem;color:#94a3b8;"><span style="color:#fff;font-weight:600;">Orchestrator</span> — Routes request to the correct specialized agent based on feature selection.</div>
        </div>
        <div style="background:#1a2235;border-radius:8px;padding:12px;display:flex;align-items:center;gap:12px;">
          <div style="width:32px;height:32px;background:rgba(0,255,157,.15);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#00ff9d;font-size:.85rem;font-weight:700;flex-shrink:0;">3</div>
          <div style="font-size:.8rem;color:#94a3b8;"><span style="color:#fff;font-weight:600;">Agent Prompt</span> — The agent constructs a domain-specific engineering prompt.</div>
        </div>
        <div style="background:#1a2235;border-radius:8px;padding:12px;display:flex;align-items:center;gap:12px;">
          <div style="width:32px;height:32px;background:rgba(255,184,0,.15);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#ffb800;font-size:.85rem;font-weight:700;flex-shrink:0;">4</div>
          <div style="font-size:.8rem;color:#94a3b8;"><span style="color:#fff;font-weight:600;">IBM Granite Inference</span> — generate_response() calls watsonx.ai for AI analysis.</div>
        </div>
        <div style="background:#1a2235;border-radius:8px;padding:12px;display:flex;align-items:center;gap:12px;">
          <div style="width:32px;height:32px;background:rgba(0,212,255,.15);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#00d4ff;font-size:.85rem;font-weight:700;flex-shrink:0;">5</div>
          <div style="font-size:.8rem;color:#94a3b8;"><span style="color:#fff;font-weight:600;">AI Report</span> — Structured analysis returned and displayed in the dashboard.</div>
        </div>
      </div>
    </div>
  </div>

  <div class="col-12">
    <div class="ss-card">
      <div class="section-title">Use Cases &amp; Applications</div>
      <div class="row g-3">
        <div class="col-md-4">
          <div style="background:#1a2235;border-radius:8px;padding:14px;">
            <div style="font-size:.82rem;font-weight:600;color:#00d4ff;margin-bottom:8px;"><i class="bi bi-building me-2"></i>Telecommunications</div>
            <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin:0;">Network engineers monitoring LTE/5G base stations, microwave backhaul links, fiber infrastructure, and enterprise WiFi deployments.</p>
          </div>
        </div>
        <div class="col-md-4">
          <div style="background:#1a2235;border-radius:8px;padding:14px;">
            <div style="font-size:.82rem;font-weight:600;color:#7c5cd8;margin-bottom:8px;"><i class="bi bi-mortarboard me-2"></i>Education &amp; Research</div>
            <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin:0;">Students and researchers studying signal processing, wireless communications, RF engineering, and AI-powered network management systems.</p>
          </div>
        </div>
        <div class="col-md-4">
          <div style="background:#1a2235;border-radius:8px;padding:14px;">
            <div style="font-size:.82rem;font-weight:600;color:#00ff9d;margin-bottom:8px;"><i class="bi bi-cpu me-2"></i>IBM AI Showcases</div>
            <p style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin:0;">Demonstrations of Agentic AI, IBM Granite Models, watsonx.ai integration, and multi-agent orchestration in technical exhibitions and hackathons.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""


# =============================================================================
# HELPER: Render page with layout
# =============================================================================
def render_page(page_title: str, active_page: str, content: str):
    """Render a page using the BASE_HTML layout template."""
    return render_template_string(
        BASE_HTML,
        page_title=page_title,
        active_page=active_page,
        content=content
    )

# =============================================================================
# FLASK ROUTES – Page Views
# =============================================================================

@app.route("/")
def home():
    """Dashboard home page — overview and agent cards."""
    return render_page("Dashboard — SignalSense AI", "home", HOME_CONTENT)

@app.route("/analysis")
def analysis():
    """Signal Analysis page — Agent 1."""
    return render_page("Signal Analysis — Agent 1", "analysis", ANALYSIS_CONTENT)

@app.route("/detection")
def detection():
    """Signal Issue Detection page — Agent 2."""
    return render_page("Issue Detection — Agent 2", "detection", DETECTION_CONTENT)

@app.route("/optimization")
def optimization():
    """Signal Optimization Advisor page — Agent 3."""
    return render_page("Optimization Advisor — Agent 3", "optimization", OPTIMIZATION_CONTENT)

@app.route("/predictive")
def predictive():
    """Predictive Analytics page — Agent 4."""
    return render_page("Predictive Analytics — Agent 4", "predictive", PREDICTIVE_CONTENT)

@app.route("/about")
def about():
    """About page — architecture and IBM watsonx.ai integration details."""
    return render_page("About — SignalSense AI", "about", ABOUT_CONTENT)

# =============================================================================
# FLASK ROUTES – API Endpoints (called via JavaScript fetch)
# =============================================================================

@app.route("/api/analysis", methods=["POST"])
def api_analysis():
    """
    API endpoint for Agent 1 — Signal Analysis.
    Accepts JSON payload with signal metrics and optional CSV data.
    Routes to signal_analysis_agent() via the orchestrator.
    """
    data = request.get_json(force=True) or {}

    # If CSV data was uploaded, append it to the historical field
    csv_data = data.get("csv_data", "").strip()
    if csv_data:
        try:
            reader = csv.DictReader(io.StringIO(csv_data))
            rows = list(reader)
            if rows:
                csv_summary = f"\nUploaded CSV Dataset ({len(rows)} rows):\n"
                # Summarize first 10 rows for prompt
                for row in rows[:10]:
                    csv_summary += str(dict(row)) + "\n"
                data["historical"] = (data.get("historical", "") + csv_summary).strip()
        except Exception:
            pass  # If CSV parsing fails, proceed with what we have

    # ---- Orchestrator routes to Agent 1 ----
    result = orchestrator("analysis", data)
    return jsonify({"result": result})

@app.route("/api/detection", methods=["POST"])
def api_detection():
    """
    API endpoint for Agent 2 — Signal Issue Detection.
    Routes to signal_issue_detection_agent() via the orchestrator.
    """
    data = request.get_json(force=True) or {}
    # ---- Orchestrator routes to Agent 2 ----
    result = orchestrator("detection", data)
    return jsonify({"result": result})

@app.route("/api/optimization", methods=["POST"])
def api_optimization():
    """
    API endpoint for Agent 3 — Optimization Advisor.
    Routes to signal_insight_agent() via the orchestrator.
    """
    data = request.get_json(force=True) or {}
    # ---- Orchestrator routes to Agent 3 ----
    result = orchestrator("insight", data)
    return jsonify({"result": result})

@app.route("/api/predictive", methods=["POST"])
def api_predictive():
    """
    API endpoint for Agent 4 — Predictive Failure Analytics.
    Accepts JSON payload including historical data and optional CSV.
    Routes to predictive_failure_agent() via the orchestrator.
    """
    data = request.get_json(force=True) or {}

    # If CSV data was uploaded, append it to historical_data
    csv_data = data.get("csv_data", "").strip()
    if csv_data:
        try:
            reader = csv.DictReader(io.StringIO(csv_data))
            rows = list(reader)
            if rows:
                csv_summary = f"\nUploaded Historical CSV ({len(rows)} rows):\n"
                for row in rows[:15]:
                    csv_summary += str(dict(row)) + "\n"
                data["historical_data"] = (data.get("historical_data", "") + csv_summary).strip()
        except Exception:
            pass

    # ---- Orchestrator routes to Agent 4 ----
    result = orchestrator("predictive", data)
    return jsonify({"result": result})

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SignalSense AI – Intelligent Signal Quality Analyzer")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print(f"  WATSONX_URL        : {WATSONX_URL}")
    print(f"  WATSONX_API_KEY    : {'SET' if WATSONX_API_KEY else 'NOT SET ⚠️'}")
    print(f"  WATSONX_PROJECT_ID : {'SET' if WATSONX_PROJECT_ID else 'NOT SET ⚠️'}")
    print("=" * 60)
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
