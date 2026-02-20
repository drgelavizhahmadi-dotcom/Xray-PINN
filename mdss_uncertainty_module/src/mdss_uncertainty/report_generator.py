"""PDF report generator for regulatory documentation."""

from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class ReportGenerator:
    """Generate AI Act supporting documentation PDF."""
    
    def __init__(
        self,
        single_case: Optional[dict] = None,
        aggregate_stats: Optional[dict] = None,
        calibration_plot_path: Optional[str] = None,
    ):
        self.single_case = single_case or {}
        self.aggregate_stats = aggregate_stats or {"n_samples": 500, "ece": 0.05}
        self.calibration_plot_path = calibration_plot_path
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        
    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name="Disclaimer",
            parent=self.styles["Normal"],
            fontSize=14,
            textColor=colors.red,
            alignment=1,
            spaceAfter=20,
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#2c5282"),
            spaceAfter=12,
        ))
        
    def generate(self, output_path: str) -> str:
        """Generate PDF report."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        
        story = []
        self._add_disclaimer(story)
        self._add_methodology(story)
        self._add_single_case(story)
        self._add_oversight_policy(story)
        self._add_aggregate_stats(story)
        self._add_regulatory_mapping(story)
        self._add_footer(story)
        
        doc.build(story)
        return output_path
        
    def _add_disclaimer(self, story: list):
        """Add red disclaimer header."""
        story.append(Paragraph(
            "DISCLAIMER: This is technical supporting documentation only, not regulatory approval",
            self.styles["Disclaimer"],
        ))
        story.append(Spacer(1, 0.5 * cm))
        
    def _add_methodology(self, story: list):
        """Add methodology section."""
        story.append(Paragraph("1. Methodology", self.styles["SectionHeader"]))
        text = """
        Bayesian Monte Carlo Dropout is applied to a DenseNet121 architecture trained on chest X-rays.
        30 stochastic forward passes generate a distribution over predictions.
        Epistemic uncertainty is quantified as the variance across Monte Carlo samples.
        """
        story.append(Paragraph(text.strip(), self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        
    def _add_single_case(self, story: list):
        """Add single case metrics table."""
        story.append(Paragraph("2. Single Case Analysis", self.styles["SectionHeader"]))
        
        data = [["Metric", "Value"]]
        data.append(["Uncertainty", f"{self.single_case.get('uncertainty', 0.0):.4f}"])
        data.append(["Confidence", f"{self.single_case.get('confidence', 0.0):.4f}"])
        data.append(["95% CI Lower", f"{self.single_case.get('ci_lower', 0.0):.4f}"])
        data.append(["95% CI Upper", f"{self.single_case.get('ci_upper', 0.0):.4f}"])
        
        table = Table(data, colWidths=[6 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3 * cm))
        
    def _add_oversight_policy(self, story: list):
        """Add oversight policy table."""
        story.append(Paragraph("3. Oversight Policy", self.styles["SectionHeader"]))
        
        data = [["Attribute", "Value"]]
        data.append(["Risk Level", self.single_case.get("risk_level", "N/A")])
        data.append(["Oversight Action", self.single_case.get("oversight_action", "N/A")])
        data.append(["Article 14 Status", self.single_case.get("article_14_status", "N/A")])
        
        table = Table(data, colWidths=[6 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3 * cm))
        
    def _add_aggregate_stats(self, story: list):
        """Add aggregate validation summary."""
        story.append(Paragraph("4. Aggregate Validation", self.styles["SectionHeader"]))
        
        n = self.aggregate_stats.get("n_samples", 500)
        ece = self.aggregate_stats.get("ece", 0.0)
        
        text = f"""
        Validation set: n={n} samples<br/>
        Expected Calibration Error (ECE): {ece:.4f}<br/>
        Calibration assessed via reliability diagram (10 bins).<br/>
        ECE < 0.1 indicates well-calibrated uncertainty estimates.
        """
        story.append(Paragraph(text.strip(), self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        
    def _add_regulatory_mapping(self, story: list):
        """Add AI Act regulatory mapping."""
        story.append(Paragraph("5. Regulatory Mapping", self.styles["SectionHeader"]))
        
        bullets = [
            "• Supports AI Act Article 15 (Accuracy): Uncertainty quantification provided",
            "• Supports AI Act Article 14 (Oversight): Risk-based human review triggers",
            "• Supports MDR Class IIa: Technical documentation for notified body review",
        ]
        for bullet in bullets:
            story.append(Paragraph(bullet, self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        
    def _add_footer(self, story: list):
        """Add limitations footer."""
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph("Limitations", self.styles["SectionHeader"]))
        
        text = """
        <i>This module quantifies epistemic uncertainty via Monte Carlo Dropout.
        Results depend on model architecture and training data distribution.
        Clinical validation by qualified radiologists is required before deployment.
        This documentation supports but does not replace formal regulatory approval.</i>
        """
        story.append(Paragraph(text.strip(), self.styles["Normal"]))
        
    def save_demo_set(self, low_case: dict, med_case: dict, high_case: dict, output_dir: str = "./demo/expected_outputs"):
        """Generate 3 demo PDFs for presentation."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for level, case in [("low", low_case), ("medium", med_case), ("high", high_case)]:
            self.single_case = case
            path = Path(output_dir) / f"demo_{level}_uncertainty.pdf"
            self.generate(str(path))
            print(f"Generated: {path}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_case = {
        "uncertainty": 0.12,
        "confidence": 0.85,
        "ci_lower": 0.78,
        "ci_upper": 0.92,
        "risk_level": "MEDIUM",
        "oversight_action": "Encourage double reading",
        "article_14_status": "Radiologist notification recommended",
    }
    
    gen = ReportGenerator(single_case=dummy_case, aggregate_stats={"n_samples": 500, "ece": 0.08})
    gen.generate("./test_report.pdf")
    print("Test report generated: ./test_report.pdf")
