"""Standalone, Cyrillic-font PDF from the generated empirical report specification."""
import argparse
from html import escape
import importlib.util
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether


def main(font_dir=None):
    if font_dir is None:
        spec=importlib.util.find_spec("matplotlib")
        if spec is None:
            raise ValueError("Provide --font-dir with DejaVuSans.ttf and DejaVuSans-Bold.ttf")
        font_dir=Path(spec.origin).parent/"mpl-data/fonts/ttf"
    for name,file in [("Research","DejaVuSans.ttf"),("ResearchBold","DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name,str(Path(font_dir)/file)))
    navy=colors.HexColor("#163047");teal=colors.HexColor("#0b7c86");gray=colors.HexColor("#526779")
    styles={
        "title":ParagraphStyle("title",fontName="ResearchBold",fontSize=25,leading=32,textColor=navy,spaceAfter=18),
        "heading":ParagraphStyle("heading",fontName="ResearchBold",fontSize=18,leading=24,textColor=navy,spaceAfter=14,keepWithNext=True),
        "body":ParagraphStyle("body",fontName="Research",fontSize=9.3,leading=13.6,textColor=navy,spaceAfter=8),
        "caption":ParagraphStyle("caption",fontName="Research",fontSize=8,leading=11,textColor=gray,spaceBefore=5,spaceAfter=10),
        "cell":ParagraphStyle("cell",fontName="Research",fontSize=7.7,leading=10.6,textColor=navy),
        "head":ParagraphStyle("head",fontName="ResearchBold",fontSize=7.7,leading=10.6,textColor=colors.white),
        "eyebrow":ParagraphStyle("eyebrow",fontName="ResearchBold",fontSize=9,leading=13,textColor=teal,spaceAfter=10)}
    def p(text,style="body"):
        return Paragraph(escape(str(text)).replace("\n","<br/>"),styles[style])
    width=A4[0]-84
    story=[]
    sections=json.loads(Path("output/report_sections.json").read_text())
    for index,section in enumerate(sections):
        if index==1 or section["title"].startswith("11."):
            story.append(PageBreak())
        elif index>1:
            story.append(Spacer(1,16))
        if index==0:
            story.append(p("ИССЛЕДОВАНИЕ / РЕКОМЕНДАТЕЛЬНЫЕ СИСТЕМЫ","eyebrow"))
        story.append(p(section["title"],"title" if index==0 else "heading"))
        story.extend(p(t) for t in section["paragraphs"])
        for tab in section["tables"]:
            n=len(tab["headers"])
            first=min(155,width*.33)
            widths=[first]+[(width-first)/(n-1)]*(n-1) if n>1 else [width]
            values=[[p(t,"head") for t in tab["headers"]]]+[[p(t,"cell") for t in row] for row in tab["rows"]]
            table=Table(values,colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),navy),("VALIGN",(0,0),(-1,-1),"TOP"),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#edf5f6"),colors.white]),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#cedde3"))]))
            story.extend([Spacer(1,5),table,Spacer(1,10)])
        for figure in section["figures"]:
            image=Image(figure["path"])
            maximum_height=620 if image.imageHeight>image.imageWidth else 330
            factor=min(width/image.imageWidth,maximum_height/image.imageHeight)
            image.drawWidth=image.imageWidth*factor;image.drawHeight=image.imageHeight*factor
            story.append(KeepTogether([image,p(figure["caption"],"caption")]))
    target=Path("output/pdf/yambda_gpt2_rl_research.pdf");target.parent.mkdir(parents=True,exist_ok=True)
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFillColor(gray);canvas.setFont("Research",7)
        canvas.drawString(42,A4[1]-25,"YAMBDA  /  GPT-2 + REINFORCE  /  24.10.2026")
        canvas.drawString(42,22,"Полный код, протокол и результаты: task2_rl")
        canvas.drawRightString(A4[0]-42,22,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=45,bottomMargin=40,
        title="RL для последовательных рекомендаций: Yambda, GPT-2, REINFORCE",author="Исследование по рекомендательным системам")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(str(target.resolve()))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--font-dir")
    main(parser.parse_args().font_dir)
