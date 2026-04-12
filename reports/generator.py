import os
import pandas as pd
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ReportGenerator:
    """
    Handles generation of forensic reports (PDF/CSV) from system outcomes.
    Used for audit trails and regulatory compliance.
    """

    @staticmethod
    def generate_csv(fault_log: list, output_path: str):
        """Generates a raw CSV dump of detected faults."""
        if not fault_log:
            df = pd.DataFrame(columns=["timestamp", "fault_type", "severity", "estimated_distance_m", "status"])
        else:
            df = pd.DataFrame(fault_log)
        
        df.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def generate_pdf(fault_log: list, metadata: dict, output_path: str):
        """Generates a professional PDF forensic report."""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # ── Header ────────────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=22,
            spaceAfter=20,
            textColor=colors.hexColor("#020912")
        )
        elements.append(Paragraph("Undersea Cable Forensic Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 20))

        # ── Metadata Section ──────────────────────────────────────────────────
        elements.append(Paragraph("System Context", styles['Heading2']))
        meta_data = [
            ["Deployment ID", metadata.get("deployment_id", "N/A")],
            ["Dataset Source", metadata.get("source", "Simulation")],
            ["Model Version", metadata.get("model_version", "v1.2.0")],
            ["Anomaly Threshold", f"{metadata.get('threshold', 0):.6f}"],
            ["Total Samples", str(metadata.get("total_samples", 0))],
            ["Total Faults", str(len(fault_log))]
        ]
        t_meta = Table(meta_data, colWidths=[150, 300])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.hexColor("#041322")),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(t_meta)
        elements.append(Spacer(1, 30))

        # ── Fault Log Table ───────────────────────────────────────────────────
        elements.append(Paragraph("Detection Log", styles['Heading2']))
        if not fault_log:
            elements.append(Paragraph("No faults detected during this operation window.", styles['Italic']))
        else:
            # Table Header
            data = [["Timestamp", "Type", "Severity", "Est. Distance (m)"]]
            for f in fault_log:
                data.append([
                    f.get("timestamp", "N/A"),
                    f.get("fault_type", "None").replace("_", " ").title(),
                    f.get("severity", "Medium").upper(),
                    f"{f.get('estimated_distance_m', 0):.1f}"
                ])

            t_log = Table(data, colWidths=[120, 150, 80, 120])
            t_log.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.hexColor("#00ffc8")), # Bio Green
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.hexColor("#020912")),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
            ]))
            elements.append(t_log)

        # ── Footer ────────────────────────────────────────────────────────────
        elements.append(Spacer(1, 50))
        elements.append(Paragraph("End of Report", styles['Centered']))

        doc.build(elements)
        return output_path
