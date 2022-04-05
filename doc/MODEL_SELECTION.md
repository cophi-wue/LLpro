# Model Selection

The selection of the language models / NLP tools for the LLP pipeline was primarily guided by the “Evaluation of off-the-shelf NLP tools for German” by [Ortmann, Roussel, Dipper (2019)](#ref-ortmann_evaluating_2019). We refer to the publication for details on the evaluated systems. Additionally, we perform evaluations on the same dataset for 

- latest Spacy v3.2,
- ParZu, with “improved statistic” derived from the non-free [Tüba-D/Z](https://uni-tuebingen.de/en/faculties/faculty-of-humanities/departments/modern-languages/department-of-linguistics/chairs/general-and-computational-linguistics/ressources/corpora/tueba-dz/) corpus.

![](accplot_new.png)
Overall F1 scores (for tokens and sentences) or accuracy (otherwise) for all systems at annotation level. Data reproduced from [Ortmann, Roussel, Dipper (2019)](#ref-ortmann_evaluating_2019). Systems with * were not part of said evaluation and are supplied by own evaluation runs on the same dataset. The cells of the final model selection are boxed.

On the basis of these evaluations, the models selected for the default LLP pipeline are

* Tokenization and Sentence Splitting via **NLTK** [(Bird, Klein, Loper 2009)](#ref-bird_natural_2009):

  NLTK implements tokenization and sentence splitting via the PUNKT algorithm [(Kiss, Strunk 2006)](#ref-kiss_unsupervised_2006). [Webpage](https://www.nltk.org/_modules/nltk/tokenize/punkt.html). This system gives the best performance when considering both tokenization and sentence splitting (and hence appears to perform better than, e.g., SoMaJo)


* POS tagging via **SoMeWeTa** [(Proisl 2018)](#ref-proisl_someweta_2018):

  Based on averaged structured perceptron; focuses on web and social media texts. [Github](https://github.com/tsproisl/SoMeWeTa). Latest Spacy (v3.2, de_dep_news_trf) appears to perform marginally better, but SoMeWeTa tags significantly faster, and does not require a GPU.

 
* Lemmatization and Morphological Analysis via **RNNTagger** [(Schmid 2019)](#ref-schmid_deep_2019):

  Based on LSTMs with character-based word representations and lemmatizes using an encoder-decoder system with attention. [Webpage](https://www.cis.uni-muenchen.de/~schmid/tools/RNNTagger/). Overall best performance in both lemmatization and morphological analysis.


* Dependency Parsing via **ParZu**, trained on TüBa-D/Z ([Sennrich, Schneider, Volk, Warin 2009](#ref-sennrich_new_2009); [Sennrich, Volk, Schneider 2013](#ref-sennrich_exploiting_2013)):

  Hybrid architecture using hand-written rules and statistical disambiguation to support a core CYK parser.
  [Github](https://github.com/rsennrich/ParZu). Only marginally better than Spacy (v3.2, de_dep_news_trf), but significantly slower due to the parser being implemented in Prolog. The default pipeline increases the speed by parallelizing on multiple cores.

## References

<div id="ref-bird_natural_2009" class="csl-entry" role="doc-biblioentry">

Bird, Steven, Ewan Klein, and Edward Loper. 2009. <em>Natural Language Processing with Python</em>. Cambridge, Mass.: O’Reilly.

</div>
<div id="ref-kiss_unsupervised_2006" class="csl-entry" role="doc-biblioentry">

Kiss, Tibor, and Jan Strunk. 2006. <span>“Unsupervised Multilingual Sentence Boundary Detection.”</span> <em>Computational Linguistics</em> 32 (4): 485–525. <a href="https://doi.org/10.1162/coli.2006.32.4.485">https://doi.org/10.1162/coli.2006.32.4.485</a>.

</div>
<div id="ref-ortmann_evaluating_2019" class="csl-entry" role="doc-biblioentry">

Ortmann, Katrin, A. Roussel, and Stefanie Dipper. 2019. <span>“Evaluating Off-the-Shelf <span>NLP</span> Tools for German.”</span> In <em>Proceedings of the 15th Conference on Natural Language Processing (<span>KONVENS</span> 2019)</em>, 212–22. Erlangen, Germany: German Society for Computational Linguistics &amp; Language Technology. <a href="https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_55.pdf">https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_55.pdf</a>.

</div>
<div id="ref-proisl_someweta_2018" class="csl-entry" role="doc-biblioentry">

Proisl, Thomas. 2018. <span>“<span>SoMeWeTa</span>: A Part-of-Speech Tagger for German Social Media and Web Texts.”</span> In <em>Proceedings of the Eleventh International Conference on Language Resources and Evaluation (<span>LREC</span> 2018)</em>, 665–70. Miyazaki, Japan: European Language Resources Association <span>ELRA</span>. <a href="http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf">http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf</a>.

</div>
<div id="ref-schmid_deep_2019" class="csl-entry" role="doc-biblioentry">

Schmid, Helmut. 2019. <span>“Deep Learning-Based Morphological Taggers and Lemmatizers for Annotating Historical Texts.”</span> In <em><span>DATeCH</span>, Proceedings of the 3rd International Conference on Digital Access to Textual Cultural Heritage</em>, 133–37. Brussels, Belgium: Association for Computing Machinery. <a href="https://www.cis.uni-muenchen.de/~schmid/papers/Datech2019.pdf">https://www.cis.uni-muenchen.de/~schmid/papers/Datech2019.pdf</a>.

</div>
<div id="ref-sennrich_new_2009" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, G. Schneider, M. Volk, M. Warin, C. Chiarcos, Richard Eckart de Castilho, and Manfred Stede. 2009. <span>“A New Hybrid Dependency Parser for German.”</span> In <em>Proceedings of the <span>GSCL</span> Conference</em>. Potsdam, Germany. <a href="https://doi.org/10.5167/UZH-25506">https://doi.org/10.5167/UZH-25506</a>.

</div>
<div id="ref-sennrich_exploiting_2013" class="csl-entry" role="doc-biblioentry">

Sennrich, Rico, Martin Volk, and Gerold Schneider. 2013. <span>“Exploiting Synergies Between Open Resources for German Dependency Parsing, <span>POS</span>-Tagging, and Morphological Analysis.”</span> In <em>Proceedings of the International Conference Recent Advances in Natural Language Processing <span>RANLP</span> 2013</em>, 601–9. Hissar, Bulgaria: <span>INCOMA</span> Ltd. Shoumen, <span>BULGARIA</span>. <a href="https://www.aclweb.org/anthology/R13-1079">https://www.aclweb.org/anthology/R13-1079</a>.

</div>

