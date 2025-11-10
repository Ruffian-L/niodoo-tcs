#!/usr/bin/env python3
"""
Generate formatted PDF research paper from NIODOO markdown
Creates a professional LaTeX-style PDF with figures and tables
"""

import sys
import re
from pathlib import Path

def escape_latex(text):
    """Escape special LaTeX characters"""
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}',
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text

def markdown_to_latex(md_content):
    """Convert markdown to LaTeX format"""
    
    # Process headers
    md_content = re.sub(r'^##\s+(.+)$', r'\\section{\1}', md_content, flags=re.MULTILINE)
    md_content = re.sub(r'^###\s+(.+)$', r'\\subsection{\1}', md_content, flags=re.MULTILINE)
    md_content = re.sub(r'^####\s+(.+)$', r'\\subsubsection{\1}', md_content, flags=re.MULTILINE)
    
    # Process bold
    md_content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', md_content)
    
    # Process code blocks
    md_content = re.sub(r'```(\w+)?\n(.*?)```', r'\\begin{verbatim}\2\\end{verbatim}', md_content, flags=re.DOTALL)
    
    # Process inline code
    md_content = re.sub(r'`([^`]+)`', r'\\texttt{\1}', md_content)
    
    # Process lists
    md_content = re.sub(r'^- (.+)$', r'\\item \1', md_content, flags=re.MULTILINE)
    
    # Process figure references
    md_content = re.sub(r'!\[([^\]]+)\]\(([^)]+)\)', r'\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.85\\textwidth]{\2}\n\\caption{\1}\n\\end{figure}', md_content)
    
    return md_content

def create_latex_document():
    """Create complete LaTeX document"""
    
    # Read markdown
    md_path = Path("NIODOO_RESEARCH_PAPER.md")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Extract abstract
    abstract_match = re.search(r'## Abstract\n\n(.*?)\n\n##', md_content, re.DOTALL)
    abstract = abstract_match.group(1) if abstract_match else ""
    
    # Extract main content (after abstract)
    main_content = re.sub(r'^## Abstract\n\n.*?\n\n##', '##', md_content, flags=re.DOTALL)
    
    # Convert to LaTeX
    latex_content = markdown_to_latex(main_content)
    
    # Create full LaTeX document
    latex_doc = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{geometry}}
\\usepackage{{float}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{enumitem}}

\\geometry{{margin=1in}}

\\title{{NIODOO: A Topological Data Analysis Framework for Adaptive AI Consciousness Simulation}}
\\author{{The NIODOO Research Team}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{escape_latex(abstract)}
\\end{{abstract}}

{latex_content}

\\end{{document}}
"""
    
    # Write LaTeX file
    tex_path = Path("NIODOO_RESEARCH_PAPER_FULL.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_doc)
    
    print(f"✓ LaTeX document created: {tex_path}")
    print("  To generate PDF, run: pdflatex NIODOO_RESEARCH_PAPER_FULL.tex")
    
    return True

if __name__ == "__main__":
    create_latex_document()

