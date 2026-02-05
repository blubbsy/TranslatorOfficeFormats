"""
Custom CSS styles for a modern, sleek UI.
"""

CUSTOM_CSS = """
<style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tighten Main Container */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Modern Headings */
    h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1e293b;
    }
    h2, h3 {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #334155;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Job Card Styling (Bordered Containers) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff;
        padding: 0.5rem 1rem !important; /* Horizontal padding, tight vertical */
        margin-bottom: 0.5rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Dark mode adjustments (rough) */
    @media (prefers-color-scheme: dark) {
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #1e293b;
            border-color: #334155 !important;
        }
        h1, h2, h3 { color: #f8fafc; }
    }

    /* Reduce Gap between elements */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Buttons */
    button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s;
    }
    
    /* Make secondary buttons (actions) subtle */
    button[kind="secondary"] {
        border-color: transparent !important;
        background-color: transparent !important;
        color: #64748b !important;
    }
    button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    
    /* Primary button punchy */
    button[kind="primary"] {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Compact File Uploader */
    [data-testid="stFileUploader"] {
        padding-top: 0;
    }
    
    /* Progress bar sleek */
    .stProgress > div > div > div > div {
        height: 6px !important;
        border-radius: 3px;
    }
</style>
"""