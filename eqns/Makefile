all: fortunatebench.pdf

TEX_FILES = fortunatebench.tex vlasov.tex
APPENDICES = ap-notation.tex

fortunatebench.pdf: ${TEX_FILES} ${APPENDICES}
	pdflatex -shell-escape fortunatebench
	bibtex fortunatebench
	pdflatex -shell-escape fortunatebench
	pdflatex -shell-escape fortunatebench

clean:
	rm -rf fortunatebench.pdf *~ fortunatebench.aux fortunatebench.bbl fortunatebench.blg fortunatebench.log
