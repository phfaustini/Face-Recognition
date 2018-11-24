*************
Face Detector
*************

Este projeto usa aprendizado supervisionado para identificar quem é quem imagens de faces.

Primeiro, as faces são identificadas e cortadas nas imagens. 
Uma equalização de histograma é feita e a imagem resultante é reescalada para 512x512.
Então, é aplicado o algoritmo PCA para reduzir a dimensionalidade (pois cada pixel é um atributo).

Finalmente, diferentes algoritmos de classificação são usados.
Modelos são treinados com 70% do dataset, e os 30% restantes são usados para teste.


Ambiente usado:
-----------------
* `Python3.7 + Anaconda <https://www.anaconda.com/download/#linux>`_
* Ver `requirements.txt <requirements.txt>`_


Estrutura:
----------

* **face_detector/** - código fonte.
* **faces/** - Arquivos .jpg dentro de pastas nomeadas de A-X. Dataset usado: http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar
* **cascades/** - .xml cascades para identificar faces. See more in http://alereimondo.no-ip.org/OpenCV/34
* **relatorio.ipynb** - jupyter notebook com o relatório.
* **slides.ipynb** - jupyter notebook com os relatório em formato de slides e com o código copiado nas células.


ATENÇÃO!!!!!!!
É necessário que a pasta **faces/** tenha subpastas nomeadas de A-X. Ela deve ser baixada do link abaixo, que já fornece as imagens na estrutura de pastas necessária:

https://drive.google.com/open?id=1QmCAA0mJHP8sIXtD19d5TuucC_7ynaLw

