#!/usr/bin/env python3
"""
Generate PDF from NIODOO Research Paper markdown
Uses weasyprint or falls back to HTML generation
"""

import sys
import os
from pathlib import Path

def generate_pdf_from_markdown():
    """Generate PDF using available tools"""
    
    # Try weasyprint first
    try:
        import weasyprint
        from markdown import markdown
        
        print("Using weasyprint for PDF generation...")
        
        # Read markdown
        md_path = Path("NIODOO_RESEARCH_PAPER.md")
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add CSS styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Times New Roman', serif;
                    font-size: 11pt;
                    line-height: 1.6;
                }}
                h1 {{ font-size: 18pt; font-weight: bold; margin-top: 20pt; }}
                h2 {{ font-size: 14pt; font-weight: bold; margin-top: 15pt; }}
                h3 {{ font-size: 12pt; font-weight: bold; margin-top: 12pt; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10pt 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8pt; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                code {{ background-color: #f4f4f4; padding: 2pt 4pt; border-radius: 3pt; }}
                pre {{ background-color: #f4f4f4; padding: 10pt; border-radius: 5pt; overflow-x: auto; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        pdf_path = Path("NIODOO_RESEARCH_PAPER.pdf")
        weasyprint.HTML(string=styled_html).write_pdf(pdf_path)
        
        print(f"✓ PDF generated successfully: {pdf_path}")
        return True
        
    except ImportError:
        print("weasyprint not available, trying alternative...")
    
    # Fallback: Generate HTML for manual conversion
    try:
        from markdown import markdown
        
        md_path = Path("NIODOO_RESEARCH_PAPER.md")
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown(md_content, extensions=['tables', 'fenced_code'])
        
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>NIODOO Research Paper</title>
            <style>
                body {{ font-family: 'Times New Roman', serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
                h1 {{ font-size: 24pt; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ font-size: 18pt; margin-top: 30px; }}
                h3 {{ font-size: 14pt; margin-top: 20px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; }}
                th {{ background-color: #f2f2f2; }}
                code {{ background-color: #f4f4f4; padding: 2px 4px; }}
                pre {{ background-color: #f4f4f4; padding: 15px; overflow-x: auto; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        html_path = Path("NIODOO_RESEARCH_PAPER.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        
        print(f"✓ HTML generated: {html_path}")
        print("  Note: To convert to PDF, use:")
        print("    - Browser: Print to PDF")
        print("    - wkhtmltopdf: wkhtmltopdf NIODOO_RESEARCH_PAPER.html output.pdf")
        print("    - pandoc: pandoc NIODOO_RESEARCH_PAPER.html -o output.pdf")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = generate_pdf_from_markdown()
    sys.exit(0 if success else 1)

