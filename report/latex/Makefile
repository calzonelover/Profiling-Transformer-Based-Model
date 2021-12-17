NAME = paper

.PHONY: paper clean
all: paper

paper: $(NAME).tex
	latexmk -xelatex -bibtex $(NAME).tex
	mv *.blg ./log
	mv *.bbl ./log
	mv *.fls ./log
	mv *.aux ./log
	mv *.log ./log
	mv *.fd* ./log
	mv *.toc ./log

rolling: $(NAME).tex
	latexmk -xelatex -pvc -bibtex -interaction=nonstopmode $(NAME).tex

clean:
	\rm ./log/* paper.pdf

