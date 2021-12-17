%%%% This class file is first developed by Yoongu Kim for his PhD Thesis while at CMU and I would like to give all the credit to him. -- Rachata Ausavarungnirun

\LoadClass[11pt, oneside, letterpaper]{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Packages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% margin
\RequirePackage{geometry}
\geometry{top=1.0in, bottom=1.0in, left=1.0in, right=1.0in}

% line spacing
\RequirePackage{setspace} % \setstretch, \singlespacing, \onehalfspacing, \doublespacing

% filler text
\RequirePackage[english]{babel}
\RequirePackage{blindtext} % \blindtext

% header
\RequirePackage{fancyhdr}

% title formatting
\RequirePackage{titling} % \thetitle, \thedate, \theauthor, \droptitle
\RequirePackage{titlesec}
\titlelabel{\thetitle.\hspace{0.4em}}
\titleformat*{\section}{\bf\fontsize{15pt}{15pt}\selectfont}
\titleformat*{\subsection}{\bf\fontsize{12pt}{12pt}\selectfont}
\titleformat*{\subsubsection}{\em\fontsize{12pt}{12pt}\selectfont}

% lists
\RequirePackage{enumitem} % \begin{itemize}, \begin{enumerate}
%\setlist{partopsep=0pt, topsep=0pt, itemsep=0pt, parsep=0pt}

% graphics
\RequirePackage{graphicx} % \includegraphics

% caption
\RequirePackage[labelsep=period, labelfont=bf]{caption}
\RequirePackage{subcaption}

% tables
\RequirePackage{booktabs}   % \toprule, \midrule, \cmidrule, \bottomrule
\RequirePackage{multirow}   % \multirow, \multicolumn
\RequirePackage{tabularx}   % \tabularx


% text
\RequirePackage{indentfirst}  % indent first paragraph
\RequirePackage[table]{xcolor}

% algorithm
\RequirePackage{algorithm}  % the floater wrapper for algorithmic
\RequirePackage[noend]{algpseudocode}
\RequirePackage{algorithmicx}

% math & units
\RequirePackage[text-rm]{siunitx}
\mathchardef\mhyphen="2D

% symbols
\RequirePackage{stmaryrd}

% fonts
\RequirePackage[scaled]{helvet}    % helvetica is larger than times
\newcommand{\hel}{\fontfamily{phv}\selectfont}

% table of contents
\RequirePackage[nottoc,numbib]{tocbibind}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliography
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\bibliographystyle{abbrv}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fonts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\RequirePackage{fontspec}
%\setmainfont[Ligatures=TeX]{Times New Roman}

%\RequirePackage{ebgaramond}
%\newfontfamily{\smallcaps}[RawFeature={+c2sc,+scmp}]{EB Garamond}
%\setromanfont[Numbers=OldStyle, Ligatures={Common}]{EB Garamond}
%\setsansfont[Scale=MatchLowercase, BoldFont={Lato Bold}]{Lato Regular}
%\setmonofont[Scale=MatchLowercase]{Source Code Pro}
