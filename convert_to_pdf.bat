@echo off
echo Converting Markdown to PDF...
pandoc "Prostate_Cancer_Prediction_Project_Guide.md" -o "Prostate_Cancer_Prediction_Project_Guide.pdf" --pdf-engine=wkhtmltopdf
echo Conversion complete!
pause