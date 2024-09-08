Welcome! 

Applied machine learning utilising vast amounts of data has aided in pattern identification, predictive analytics, and solving complex problems across a multitude of fields. Solving these complex problems within these fields, researchers would find differing answers to the following questions; what machine learning techniques can we apply to the problem, how do we apply the techniques in the context of this field, and why do we need to apply them in this way? In any case, applied machine learning requires an interdisciplinary understanding of computing techniques and the field in question.

The aim of this project is to provide you with a set of reproducible tutorials that include all necessary data, code, and descriptions to replicate key results, along with demonstrations of common pitfalls, in the field of genomics. It is designed for users with knowledge of machine learning but little or no background in biology as a process to learn about applying machine learning techniques in genomics.

The Markdown book is written in R markdown, hosted by Github Pages and is available at:

https://sn2023imperial.github.io/genomicsmarkdown/

If you decide to download it locally you can use the following steps to create a PDF copy using R:

install.packages("bookdown")

install.packages("reticulate")

library(reticulate)
<br>
(for browser)
bookdown::serve_book() 
<br>
(for pdf)
rmarkdown::render_site(encoding = 'UTF-8')
<br>

The tutorials are available online through the markdown book. However to access them locally, pull the Github Repository and follow these steps:

(create venv)

python3 -m venv venv

source venv/bin/activate

pip install from requirements.txt

pip install -r requirements.txt