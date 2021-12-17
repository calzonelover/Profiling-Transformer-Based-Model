\LoadClass[times, 10pt, onecolumn]{article}
% Compress vertical space around section and subsection headings. Make
% subsubsection headings run-in (i.e. part of the following paragraph).
\RequirePackage[compact]{titlesec}
\titleformat{\section}[hang]{\normalfont\large\bfseries}{\thesection. }%
  {0pt}{}{}
\titleformat{\subsection}[hang]{\normalfont\bfseries}{\thesubsection. }%
  {0pt}{}{}
\titleformat{\subsubsection}[runin]{\normalfont\bfseries}{\thesubsubsection. }%
  {0pt}{}{}
\newlength{\sectionspace}
\newlength{\subsectionspace}
\setlength{\sectionspace}{0ex plus 0.2ex}
\setlength{\subsectionspace}{1ex plus 0.25ex}
\titlespacing*{\section}{0pt}{\sectionspace}{\sectionspace}
\titlespacing*{\subsection}{0pt}{\subsectionspace}{\subsectionspace}
\titlespacing*{\subsubsection}{0pt}{\parskip}{0.2em plus 0.1em minus 0.05em}
\setlength{\textfloatsep}{0.5em plus 0.2em minus 0.2em}

