"""
User interface for the RFQ Cybersecurity app.

This module defines the Streamlit UI that users interact with.
It leverages the services and utilities defined elsewhere in the
package to perform PDF analysis and report generation.  Keeping
the UI in its own module separates presentation from business
logic and makes it easier to iterate on the user experience.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Dict, List

import streamlit as st

# Import the package logger
from ..logging_config import get_logger

# Create a logger for the UI module
logger = get_logger(__name__)

from ..services.report_service import generate_report_parallel
from ..utils.report_utils import json_to_docx, load_header_image


def _get_heading_list() -> List[str]:
    """Return the list of cybersecurity analysis topics.

    The list is defined here to centralise changes to the topics
    without cluttering the main UI function.
    """
    return [
        "IEC 62443",
        "Perdue model",
        "Windows patches",
        "Antivirus/malware",
        "Whitelisting / allow listing",
        "Device control",
        "Firewall",
        "IDS",
        "DPI/Deep pack inspection",
        "Next Gen firewall/NG FW",
        "Network monitoring/Network intrusion detection",
        "Event monitoring/windows events/incident reporting",
        "Data diode",
        "Endpoint",
        "Risk assessment/GAP Analysis",
        "Incident response",
        "SIEM",
        "Disaster recovery",
        "Backup",
        "Network Management",
        "Virtualization/ESXI",
        "Remote Access System",
        "Asset Inventory",
        "Active Directory/IAM",
        "RBAC",
        "System/Network Hardening",
        "Security Level",
        "DMZ (Demilitarized Zone)",
        "VLAN",
    ]


def run_app() -> None:
    """Render the Streamlit user interface and handle user interactions."""
    # Configure the page title and layout.  Setting page config must
    # happen at the very beginning of the function before any widgets
    # are rendered.  ``initial_sidebar_state`` controls the sidebar visibility.
    st.set_page_config(
        page_title="Cybersecurity Report Generator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject custom CSS for a polished look
    # Apply custom CSS styles to adjust container widths, paddings and colours
    st.markdown(
        """
        <style>
            .report-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 1rem;
            }
            .heading-list {
                font-size: 0.9rem;
                line-height: 1.4;
            }
            .summary-section {
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                background-color: #fafafa;
            }
            .summary-section h4 {
                margin-top: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Attempt to load and display a decorative header image if present
    # Compute the path relative to the package root
    # Attempt to locate the decorative header image relative to the package root
    package_root =r"C:\Users\INNIMK\OneDrive - ABB\Documents\ABB Projects\RFQ Digital\RFQ Digital-1\rfq_digital_cybersecurity_roopa\rfq_cybersecurity\ui"
    header_image_path = os.path.join(package_root, "Neon Web of Digital Connectivity.png")
    if os.path.exists(header_image_path):
        # Convert the image file into a base64 data URI for embedding in HTML
        header_data_uri = load_header_image(header_image_path)
        st.markdown(
            f"""
            <div style="background-image: url('{header_data_uri}');
                        background-size: cover;
                        border-radius: 8px;
                        height: 200px;
                        margin-bottom: 1rem;
                        display: flex;
                        align-items: flex-end;
                        padding: 1rem;">
                <h1 style="color: white; margin: 0;">Cybersecurity Report Generator</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback title if the image does not exist
        st.title("Cybersecurity Report Generator")
    # Display the heading list in the sidebar
    # Retrieve the static list of headings for analysis
    heading_list = _get_heading_list()
    st.sidebar.header("Analysis Topics")
    st.sidebar.markdown(
        """<div class="heading-list">""" + "<br>".join(heading_list) + "</div>",
        unsafe_allow_html=True,
    )
    # File uploader for the PDF
    # Provide an upload widget that accepts PDF files only
    pdf_file = st.file_uploader(
        label="Upload PDF Document",
        type=["pdf"],
        help="Provide the specification or policy document you wish to analyse.",
    )
    # When the user uploads a file, show a button to run the analysis
    if pdf_file is not None:
        # Persist the uploaded file to a temporary location because PyPDFLoader expects a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Write the uploaded PDF bytes to the temporary file
            tmp_file.write(pdf_file.getvalue())
            tmp_pdf_path = tmp_file.name
        # Button that triggers report generation
        if st.button("Generate Report"):
            # Display a progress bar while processing
            progress_bar = st.progress(0.0, text="Preparing vector store and analysing sections…")
            def update_progress(fraction: float) -> None:
                # Update the progress bar and text as each heading completes
                progress_bar.progress(fraction, text="Analysing headings…")
            try:
                with st.spinner("Please wait while the report is generated…"):
                    # Generate the report using the service layer.  If any
                    # exception is raised, it will be caught below.
                    result = generate_report_parallel(tmp_pdf_path, heading_list, progress_callback=update_progress)
            except Exception as exc:
                # Log the exception and show a user‑friendly error message
                logger.error("Report generation failed: %s", exc)
                st.error(f"An error occurred during report generation: {exc}")
                return
            # Save the JSON result to a temporary directory (not shown in UI)
            json_filename = f"extracted_results_{os.path.basename(pdf_file.name).rsplit('.', 1)[0]}.json"
            json_path = os.path.join(tempfile.gettempdir(), json_filename)
            try:
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning("Failed to write JSON result file '%s': %s", json_path, exc)
            # Create a Word document from the results
            try:
                doc = json_to_docx(result)
                doc_stream = io.BytesIO()
                doc.save(doc_stream)
                doc_stream.seek(0)
            except Exception as exc:
                logger.error("Failed to create Word document: %s", exc)
                st.error(f"Could not create Word document: {exc}")
                return
            st.success("Report generation complete!")
            # Display the results in the main panel using expanders
            for heading, sections in result.items():
                if not sections:
                    continue
                with st.expander(f"{heading}"):
                    for section_name, snippet in sections.items():
                        st.markdown(f"**{section_name}**")
                        st.write(snippet)
            # Provide a download button for the Word file
            st.download_button(
                label="Download Word Report",
                data=doc_stream.getvalue(),
                file_name=f"{os.path.splitext(pdf_file.name)[0]}_cybersecurity_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )