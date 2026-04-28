# PowerScript to convert Markdown to PDF using Pandoc
# Make sure Pandoc is installed first

Write-Host "Converting Markdown to PDF..." -ForegroundColor Green

# Basic conversion
pandoc "Prostate_Cancer_Prediction_Project_Guide.md" -o "Prostate_Cancer_Prediction_Project_Guide.pdf"

# Enhanced conversion with better formatting
# pandoc "Prostate_Cancer_Prediction_Project_Guide.md" -o "Prostate_Cancer_Prediction_Project_Guide.pdf" --pdf-engine=wkhtmltopdf --css=style.css --toc --number-sections

Write-Host "Conversion completed!" -ForegroundColor Green
Write-Host "PDF saved as: Prostate_Cancer_Prediction_Project_Guide.pdf" -ForegroundColor Yellow