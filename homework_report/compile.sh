#!/bin/bash

xelatex anliang.tex
bibtex anliang.aux
xelatex anliang.tex 
xelatex anliang.tex 

