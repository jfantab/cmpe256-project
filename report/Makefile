all:
	pdflatex report.tex

bib:
	pdflatex report.tex
	bibtex report
	pdflatex report.tex

bibtex:
	pdflatex report.tex
	bibtex report
	pdflatex report.tex

clean:
	rm -f report.bbl report.blg report.out report.aux report.log

